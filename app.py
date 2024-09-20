from flask import Flask, render_template, request, jsonify
from astropy.io import fits
import numpy as np
import json
import plotly.graph_objs as go
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor
import logging
from io import BytesIO
import yaml

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Updated color scales to match CSS
COLOR_SCALES = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Viridis', 'Spectral', 'RdYlBu', 'Picnic']


def load_config(config_file='config.yaml'):
    """Load configuration from a YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading configuration: {str(e)}. Using default values.")
        return {}


CONFIG = load_config()
DATA_DIR = CONFIG.get('data_dir', 'Data')


def load_fits(file, memmap=True):
    try:
        file_content = file.read()
        file_like = BytesIO(file_content)
        with fits.open(file_like, memmap=memmap, mode='readonly') as hdul:
            if memmap:
                # If memmapping, return a copy of the data to ensure it's loaded into memory
                return np.array(hdul[0].data)
            else:
                return hdul[0].data
    except Exception as e:
        logger.error(f"Error loading FITS file: {str(e)}")
        raise


def calculate_bin_size(data_length, num_plots):
    """Calculate the bin size based on the data length and desired number of plots."""
    return max(1, data_length // num_plots)


def bin_flux_arr(fluxarr, bin_size):
    """
    Bin the flux array using a specified bin size. Uses vectorized operations.
    """
    try:
        n_bins = fluxarr.shape[1] // bin_size
        bin_edges = np.linspace(0, fluxarr.shape[1], n_bins + 1)

        def bin_row(row):
            return binned_statistic(np.arange(len(row)), row, statistic='median', bins=bin_edges)[0]

        with ThreadPoolExecutor() as executor:
            fluxarrbin = np.array(list(executor.map(bin_row, fluxarr)))

        return fluxarrbin
    except Exception as e:
        logger.error(f"Error in bin_flux_arr: {str(e)}")
        raise


def smooth_flux(flux, sigma=2):
    """
    Apply Gaussian smoothing to the flux data.
    """
    try:
        return gaussian_filter(flux, sigma=sigma)
    except Exception as e:
        logger.error(f"Error in smooth_flux: {str(e)}")
        raise


def process_data(flux, wavelength, time, num_plots, remove_first_60=True, apply_binning=True,
                 smooth_sigma=2, wavelength_unit='um'):
    """
    Process the flux, wavelength, and time data for plotting.
    """
    try:
        logger.info('Shape before processing: %s', flux.shape)

        # Ensure wavelength and flux are compatible
        min_length = min(flux.shape[0], len(wavelength))
        flux = flux[:min_length]
        wavelength = wavelength[:min_length]

        # Calculate bin size based on the number of plots requested
        bin_size = calculate_bin_size(flux.shape[1], num_plots)
        logger.info(f'Calculated bin size: {bin_size}')

        if bin_size > 1 and apply_binning:
            flux = bin_flux_arr(flux, bin_size)
            logger.info('Shape after binning: %s', flux.shape)

        flux = smooth_flux(flux, sigma=smooth_sigma)
        logger.info('Shape after smoothing: %s', flux.shape)

        # Convert wavelength units if necessary
        if wavelength_unit == 'nm':
            wavelength = wavelength / 1000
            wavelength_label = 'Wavelength (nm)'
        elif wavelength_unit == 'A':
            wavelength = wavelength / 10000
            wavelength_label = 'Wavelength (Å)'
        else:
            wavelength_label = 'Wavelength (µm)'

        x = np.linspace(0, 1, flux.shape[1]) * ((np.nanmax(time) - np.nanmin(time)) * 24.)
        y = wavelength[60:] if remove_first_60 else wavelength
        X, Y = np.meshgrid(x, y)
        Z = (flux[60:] - 1) * 100 if remove_first_60 else (flux - 1) * 100

        return x, y, X, Y, Z, wavelength_label

    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise


def create_surface_plot(flux, wavelength, time, title, num_plots, remove_first_60=True, apply_binning=True,
                        smooth_sigma=2, wavelength_unit='um', custom_bands=None, colorscale='Viridis'):
    """
    Create an enhanced 3D surface plot with masking for custom bands.
    """
    x, y, X, Y, Z, wavelength_label = process_data(
        flux, wavelength, time, num_plots, remove_first_60, apply_binning,
        smooth_sigma, wavelength_unit
    )

    hovertemplate = ('Time: %{x:.2f} hours<br>' +
                     wavelength_label + ': %{y:.4f}<br>' +
                     'Variability: %{z:.4f}%<br>' +
                     '<extra></extra>')

    surface_full = go.Surface(
        x=X, y=Y, z=Z,
        colorscale=colorscale,
        opacity=1,
        name='Full Spectrum',
        colorbar=dict(
            title='Variability %',
            titleside='right',
            titlefont=dict(size=12, color='#ffffff'),
            tickfont=dict(size=10, color='#ffffff'),
            len=0.8,
            thickness=15,
            x=1.0
        ),
        hoverinfo='x+y+z',
        hovertemplate=hovertemplate
    )

    gray_surface = go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, 'rgba(200, 200, 200, 0.3)'], [1, 'rgba(200, 200, 200, 0.3)']],
        opacity=0.3,
        showscale=False,
        hoverinfo='skip'
    )

    data = [surface_full, gray_surface]

    if custom_bands:
        for band in custom_bands:
            band_mask = (Y >= band['start']) & (Y <= band['end'])
            band_surface = go.Surface(
                x=X, y=Y, z=np.where(band_mask, Z, np.nan),
                colorscale=colorscale,
                opacity=0.9,
                showscale=False,
                name=band['name'],
                hoverinfo='x+y+z',
                hovertemplate=hovertemplate
            )
            data.append(band_surface)

    layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=18, color='#ffffff')
        ),
        scene=dict(
            xaxis=dict(title='Time (hours)', gridcolor='#555555', linecolor='#555555', showbackground=True,
                       backgroundcolor='rgba(0,0,0,0.5)'),
            yaxis=dict(title=wavelength_label, gridcolor='#555555', linecolor='#555555', showbackground=True,
                       backgroundcolor='rgba(0,0,0,0.5)'),
            zaxis=dict(title='Variability %', gridcolor='#555555', linecolor='#555555', showbackground=True,
                       backgroundcolor='rgba(0,0,0,0.5)'),
            aspectmode='manual',
            aspectratio=dict(x=1.4, y=1.2, z=0.8),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        autosize=True,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='#ffffff', size=10)
        )
    )

    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            x=0.1,
            y=-0.05,
            xanchor="center",
            yanchor="top",
            buttons=[
                        dict(args=[{'visible': [True, False] + [False] * len(custom_bands)}],
                             label="Full Spectrum",
                             method="update"),
                    ] + [
                        dict(args=[{'visible': [False, True] + [i == j for j in range(len(custom_bands))]}],
                             label=band['name'],
                             method="update") for i, band in enumerate(custom_bands or [])
                    ],
            pad={"r": 10, "t": 10},
            showactive=True
        ),
        dict(
            type="buttons",
            direction="right",
            x=0.1,
            y=-0.15,
            xanchor="center",
            yanchor="top",
            buttons=[
                dict(args=[{'scene.camera.eye': {'x': 1.5, 'y': 1.5, 'z': 1.3}}],
                     label="Default View",
                     method="relayout"),
                dict(args=[{'scene.camera.eye': {'x': 0, 'y': 0, 'z': 2.8}}],
                     label="Top View",
                     method="relayout"),
                dict(args=[{'scene.camera.eye': {'x': 2.5, 'y': 0, 'z': 0}}],
                     label="Side View",
                     method="relayout")
            ],
            pad={"r": 10, "t": 10},
            showactive=True
        )
    ]

    layout.updatemenus = updatemenus

    fig = go.Figure(data=data, layout=layout)

    # Update button styles
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=button['args'],
                        label=button['label'],
                        method=button['method'],
                        visible=True,
                    ) for button in menu['buttons']
                ],
                direction=menu['direction'],
                showactive=True,
                type=menu['type'],
                x=menu['x'],
                xanchor=menu['xanchor'],
                y=menu['y'],
                yanchor=menu['yanchor'],
                pad={"r": 10, "t": 10},
                bgcolor='rgba(30, 40, 70, 0.8)',
                bordercolor='#ffffff',
                font=dict(color='#ffffff'),
                active=0,
            ) for menu in updatemenus
        ]
    )

    return fig


def create_heatmap_plot(flux, wavelength, time, title, num_plots, remove_first_60=True, apply_binning=True,
                        smooth_sigma=2, wavelength_unit='um', custom_bands=None, colorscale='Viridis'):
    """Create an enhanced heatmap plot in dark mode with black-and-white masking and consistent color intensity."""
    x, y, X, Y, Z, wavelength_label = process_data(
        flux, wavelength, time, num_plots, remove_first_60, apply_binning,
        smooth_sigma, wavelength_unit
    )

    hovertemplate = ('Time: %{x:.2f} hours<br>' +
                     wavelength_label + ': %{y:.4f}<br>' +
                     'Variability: %{z:.4f}%<br>' +
                     '<extra></extra>')

    heatmap_full = go.Heatmap(
        x=x,
        y=y,
        z=Z,
        colorscale=colorscale,
        colorbar=dict(
            title='Variability %',
            titleside='right',
            titlefont=dict(size=12, color='#ffffff'),
            tickfont=dict(size=10, color='#ffffff'),
            len=0.8,
            thickness=15,
            x=1.0
        ),
        hoverinfo='x+y+z',
        hovertemplate=hovertemplate,
        name='Full Spectrum'
    )

    gray_heatmap = go.Heatmap(
        x=x,
        y=y,
        z=Z,
        colorscale=[[0, 'black'], [1, 'white']],
        opacity=0.3,
        showscale=False,
        hoverinfo='skip',
        name='Gray Mask'
    )

    data = [heatmap_full, gray_heatmap]

    if custom_bands:
        for band in custom_bands:
            band_mask = (y >= band['start']) & (y <= band['end'])
            band_z = np.where(band_mask[:, None], Z, np.nan)
            band_heatmap = go.Heatmap(
                x=x,
                y=y,
                z=band_z,
                colorscale=colorscale,
                showscale=False,
                hoverinfo='x+y+z',
                hovertemplate=hovertemplate,
                name=band['name']
            )
            data.append(band_heatmap)

    layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=18, color='#ffffff')
        ),
        xaxis=dict(title='Time (hours)', gridcolor='#555555', linecolor='#555555'),
        yaxis=dict(title=wavelength_label, gridcolor='#555555', linecolor='#555555'),
        margin=dict(l=50, r=50, t=80, b=180),
        autosize=True,
        hovermode='closest',
    )

    buttons = [
        dict(
            args=[{'visible': [True, False] + [False] * len(custom_bands)}],
            label="Full Spectrum",
            method="update"
        )
    ]

    for i, band in enumerate(custom_bands):
        buttons.append(dict(
            args=[{'visible': [False, True] + [j == i for j in range(len(custom_bands))]}],
            label=band['name'],
            method="update"
        ))

    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            x=0.5,
            y=-0.5,
            xanchor="center",
            yanchor="top",
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            active=0,
            bgcolor='rgba(30, 40, 70, 0.8)',
            bordercolor='#ffffff',
            font=dict(color='#ffffff')
        )
    ]

    layout.updatemenus = updatemenus

    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(rangeslider_visible=True)

    return fig
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        flux_file = request.files['flux']
        wavelength_file = request.files['wavelength']
        time_file = request.files['time']
        num_plots = int(request.form.get('num_plots', 1000))
        colorscale = request.form.get('colorscale', 'Viridis')
        custom_bands = json.loads(request.form.get('custom_bands', '[]'))
        remove_first_60 = request.form.get('remove_first_60', 'true').lower() == 'true'
        apply_binning = request.form.get('apply_binning', 'true').lower() == 'true'
        smooth_sigma = float(request.form.get('smooth_sigma', 2))
        wavelength_unit = request.form.get('wavelength_unit', 'um')

        logger.info(f"Received files: {flux_file.filename}, {wavelength_file.filename}, {time_file.filename}")
        logger.info(f"Number of plots: {num_plots}, Colorscale: {colorscale}")
        logger.info(f"Remove first 60: {remove_first_60}, Apply binning: {apply_binning}, Smooth sigma: {smooth_sigma}")
        logger.info(f"Wavelength unit: {wavelength_unit}")
        logger.info(f"Custom bands: {custom_bands}")

        # Load FITS files using optimized function
        flux_data = load_fits(flux_file)
        wavelength_data = load_fits(wavelength_file)
        time_data = load_fits(time_file)

        logger.info(
            f"Data shapes - Flux: {flux_data.shape}, Wavelength: {wavelength_data.shape}, Time: {time_data.shape}")

        # Ensure custom bands include CH4 and CO bands if not already present
        ch4_band = {'name': 'CH₄ Band', 'start': 2.14, 'end': 2.5}
        co_band = {'name': 'CO Band', 'start': 4.5, 'end': 5.05}

        if not any(band['name'] == 'CH₄ Band' for band in custom_bands):
            custom_bands.append(ch4_band)
        if not any(band['name'] == 'CO Band' for band in custom_bands):
            custom_bands.append(co_band)

        # Use ThreadPoolExecutor for parallel processing of surface and heatmap plots
        with ThreadPoolExecutor(max_workers=2) as executor:
            surface_future = executor.submit(
                create_surface_plot,
                flux_data, wavelength_data, time_data,
                title="3D Surface Plot",
                num_plots=num_plots,
                remove_first_60=remove_first_60,
                apply_binning=apply_binning,
                smooth_sigma=smooth_sigma,
                wavelength_unit=wavelength_unit,
                custom_bands=custom_bands,
                colorscale=colorscale
            )

            heatmap_future = executor.submit(
                create_heatmap_plot,
                flux_data, wavelength_data, time_data,
                title="Heatmap",
                num_plots=num_plots,
                remove_first_60=remove_first_60,
                apply_binning=apply_binning,
                smooth_sigma=smooth_sigma,
                wavelength_unit=wavelength_unit,
                custom_bands=custom_bands,
                colorscale=colorscale
            )

            surface_plot = surface_future.result()
            heatmap_plot = heatmap_future.result()

        return jsonify({
            'surface_plot': surface_plot.to_json(),
            'heatmap_plot': heatmap_plot.to_json()
        })
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
