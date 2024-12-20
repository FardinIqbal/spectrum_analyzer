document.addEventListener('DOMContentLoaded', function() {
    // ===================================
    // Event Listeners for UI Interactions
    // ===================================
    document.getElementById('addBandBtn').addEventListener('click', () => addCustomBand());
    document.getElementById('generatePlotsBtn').addEventListener('click', generatePlots);
    document.getElementById('resetSurfaceViewBtn').addEventListener('click', () => resetPlotView('surfacePlot'));
    document.getElementById('resetHeatmapViewBtn').addEventListener('click', () => resetPlotView('heatmapPlot'));

    // ===================================
    // Additional MAST Upload Functionality
    // ===================================
    // Developer Note: The "Process MAST Folder" button will trigger uploading a .zip file
    // containing a MAST directory structure and will attempt to process all x1dints.fits files found.
    const uploadMastBtn = document.getElementById('uploadMastBtn');
    const mastZipFileInput = document.getElementById('mastZipFile');
    if (uploadMastBtn && mastZipFileInput) {
        uploadMastBtn.addEventListener('click', uploadMastDirectory);
    }

    initializeColorScales();

    // Add preset custom bands (for demonstration and consistency)
    // CH₄ and CO bands are often critical for spectral analysis and included by default.
    addCustomBand('CH₄ Band', 2.14, 2.5);
    addCustomBand('CO Band', 4.5, 5.05);
});

const colorScales = [
    { name: 'Viridis', class: 'viridis' },
    { name: 'Plasma', class: 'plasma' },
    { name: 'Inferno', class: 'inferno' },
    { name: 'Magma', class: 'magma' },
    { name: 'Cividis', class: 'cividis' },
    { name: 'Turbo', class: 'turbo' },
    { name: 'Coolwarm', class: 'coolwarm' },
    { name: 'Spectral', class: 'spectral' },
    { name: 'RdYlBu', class: 'rdylbu' },
    { name: 'Picnic', class: 'picnic' }
];

function initializeColorScales() {
    // Developer Note: Dynamically add color scale options as clickable boxes.
    const container = document.getElementById('colorscaleSelector');
    colorScales.forEach((scale, index) => {
        const option = document.createElement('div');
        option.className = `colorscale-option ${scale.class}`;
        option.setAttribute('data-colorscale', scale.name);
        option.title = scale.name;
        option.addEventListener('click', () => selectColorScale(option));
        container.appendChild(option);

        // Select the first colorscale by default
        if (index === 0) selectColorScale(option);
    });
}

function selectColorScale(selectedOption) {
    // Developer Note: Highlight the chosen colorscale visually and record it as selected.
    document.querySelectorAll('.colorscale-option').forEach(option => {
        option.classList.remove('selected');
    });
    selectedOption.classList.add('selected');
}

function addCustomBand(name = '', start = '', end = '') {
    // Developer Note: Allows adding a new custom band entry with optional initial values
    // Name: descriptive label for the band
    // Start/End: wavelength range in microns (or other chosen units)
    const bandContainer = document.createElement('div');
    bandContainer.className = 'flex items-center space-x-2 mb-2';
    bandContainer.innerHTML = `
        <input type="text" placeholder="Band Name" value="${name}" class="flex-grow px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <input type="number" step="0.01" placeholder="Start" value="${start}" class="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <input type="number" step="0.01" placeholder="End" value="${end}" class="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <button class="px-3 py-2 bg-red-600 text-gray-100 rounded-md hover:bg-red-700 transition duration-200">Remove</button>
    `;
    document.getElementById('customBands').appendChild(bandContainer);

    // Allow removal of the band entry
    bandContainer.querySelector('button').addEventListener('click', () => {
        bandContainer.remove();
    });
}

async function generatePlots() {
    // Developer Note: Gathers user input (FITS files, number of plots, custom bands),
    // then sends an AJAX POST request to the '/upload' endpoint to generate plots.
    const formData = new FormData();
    const fluxFile = document.getElementById('fluxFile').files[0];
    const wavelengthFile = document.getElementById('wavelengthFile').files[0];
    const timeFile = document.getElementById('timeFile').files[0];

    if (!fluxFile || !wavelengthFile || !timeFile) {
        alert('Please upload all required FITS files (flux, wavelength, time).');
        return;
    }

    formData.append('flux', fluxFile);
    formData.append('wavelength', wavelengthFile);
    formData.append('time', timeFile);
    formData.append('num_plots', document.getElementById('numPlots').value);
    formData.append('colorscale', document.querySelector('.colorscale-option.selected').getAttribute('data-colorscale'));

    // Extract custom band definitions from the UI
    const customBands = Array.from(document.getElementById('customBands').children).map(band => {
        const inputs = band.querySelectorAll('input');
        return {
            name: inputs[0].value.trim(),
            start: parseFloat(inputs[1].value),
            end: parseFloat(inputs[2].value)
        };
    }).filter(band => band.name && !isNaN(band.start) && !isNaN(band.end));

    formData.append('custom_bands', JSON.stringify(customBands));

    try {
        // Send data to the server to generate plots
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Plotly expects JSON-encoded figure data
        const surfaceData = JSON.parse(data.surface_plot);
        const heatmapData = JSON.parse(data.heatmap_plot);

        // Render the returned plots
        Plotly.newPlot('surfacePlot', surfaceData.data, surfaceData.layout, { responsive: true });
        Plotly.newPlot('heatmapPlot', heatmapData.data, heatmapData.layout, { responsive: true });

        // Smooth scroll to the plots container once rendered
        document.getElementById('plotsContainer').scrollIntoView({behavior: 'smooth'});
    } catch (error) {
        console.error('Error generating plots:', error);
        alert('Error generating plots: ' + error.message);
    }
}

async function uploadMastDirectory() {
    // Developer Note: Handles uploading a zipped MAST directory to the '/upload_mast' endpoint.
    // This is similar to the individual file upload but designed to handle an entire directory structure.
    const mastZipFile = document.getElementById('mastZipFile').files[0];

    if (!mastZipFile) {
        alert('Please select a MAST ZIP file before processing.');
        return;
    }

    const formData = new FormData();
    formData.append('mast_zip', mastZipFile);

    try {
        const response = await fetch('/upload_mast', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // The response should contain JSON for surface_plot and heatmap_plot
        const surfaceData = JSON.parse(data.surface_plot);
        const heatmapData = JSON.parse(data.heatmap_plot);

        // Render the MAST-based plots using Plotly
        Plotly.newPlot('surfacePlot', surfaceData.data, surfaceData.layout, { responsive: true });
        Plotly.newPlot('heatmapPlot', heatmapData.data, heatmapData.layout, { responsive: true });

        document.getElementById('plotsContainer').scrollIntoView({behavior: 'smooth'});
    } catch (error) {
        console.error('Error processing MAST folder:', error);
        alert('Error processing MAST folder: ' + error.message);
    }
}

function updatePlotLayout(plotId, updates) {
    // Developer Note: A helper function to update the layout of a Plotly plot.
    Plotly.update(plotId, {}, updates);
}

function resetPlotView(plotId) {
    // Developer Note: Reset the view of the selected plot to default values.
    if (plotId === 'surfacePlot') {
        // Reset the 3D surface plot camera to default perspective
        updatePlotLayout(plotId, {
            'scene.camera': { eye: { x: 1.5, y: 1.5, z: 1.3 } }
        });
    } else if (plotId === 'heatmapPlot') {
        // Reset the heatmap axes ranges
        Plotly.relayout(plotId, {
            'xaxis.autorange': true,
            'yaxis.autorange': true
        });
    }
}

// ===================================
// Plot Interaction Handlers
// ===================================
// Developer Note: Optionally handle plot interactions, such as camera changes or zoom/pan events.
// For instance, you might listen to 'plotly_relayout' events to respond to user navigations.

document.getElementById('surfacePlot').on('plotly_relayout', function(eventData) {
    // If the user adjusts the camera, update the layout accordingly.
    if (eventData['scene.camera']) {
        updatePlotLayout('surfacePlot', { 'scene.camera': eventData['scene.camera'] });
    }
});

document.getElementById('heatmapPlot').on('plotly_relayout', function(eventData) {
    // If the user zooms or pans on the heatmap, adjust the layout accordingly.
    if (eventData['xaxis.range[0]'] || eventData['yaxis.range[0]']) {
        updatePlotLayout('heatmapPlot', {
            'xaxis.range': [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']],
            'yaxis.range': [eventData['yaxis.range[0]'], eventData['yaxis.range[1]']]
        });
    }
});
