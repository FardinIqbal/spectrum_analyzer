document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('addBandBtn').addEventListener('click', () => addCustomBand());
    document.getElementById('generatePlotsBtn').addEventListener('click', generatePlots);
    document.getElementById('resetSurfaceViewBtn').addEventListener('click', () => resetPlotView('surfacePlot'));
    document.getElementById('resetHeatmapViewBtn').addEventListener('click', () => resetPlotView('heatmapPlot'));

    initializeColorScales();

    // Add preset custom bands
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
    const container = document.getElementById('colorscaleSelector');
    colorScales.forEach((scale, index) => {
        const option = document.createElement('div');
        option.className = `colorscale-option ${scale.class}`;
        option.setAttribute('data-colorscale', scale.name);
        option.title = scale.name;
        option.addEventListener('click', () => selectColorScale(option));
        container.appendChild(option);

        if (index === 0) selectColorScale(option);
    });
}

function selectColorScale(selectedOption) {
    document.querySelectorAll('.colorscale-option').forEach(option => {
        option.classList.remove('selected');
    });
    selectedOption.classList.add('selected');
}

function addCustomBand(name = '', start = '', end = '') {
    const bandContainer = document.createElement('div');
    bandContainer.className = 'flex items-center space-x-2 mb-2';
    bandContainer.innerHTML = `
        <input type="text" placeholder="Band Name" value="${name}" class="flex-grow px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <input type="number" step="0.01" placeholder="Start" value="${start}" class="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <input type="number" step="0.01" placeholder="End" value="${end}" class="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <button class="px-3 py-2 bg-red-600 text-gray-100 rounded-md hover:bg-red-700 transition duration-200">Remove</button>
    `;
    document.getElementById('customBands').appendChild(bandContainer);

    bandContainer.querySelector('button').addEventListener('click', () => {
        bandContainer.remove();
    });
}

async function generatePlots() {
    const formData = new FormData();
    const fluxFile = document.getElementById('fluxFile').files[0];
    const wavelengthFile = document.getElementById('wavelengthFile').files[0];
    const timeFile = document.getElementById('timeFile').files[0];

    if (!fluxFile || !wavelengthFile || !timeFile) {
        alert('Please upload all required FITS files.');
        return;
    }

    formData.append('flux', fluxFile);
    formData.append('wavelength', wavelengthFile);
    formData.append('time', timeFile);
    formData.append('num_plots', document.getElementById('numPlots').value);
    formData.append('colorscale', document.querySelector('.colorscale-option.selected').getAttribute('data-colorscale'));

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

        Plotly.newPlot('surfacePlot', JSON.parse(data.surface_plot), {
            responsive: true,
            autosize: true,
            height: 600
        });

        Plotly.newPlot('heatmapPlot', JSON.parse(data.heatmap_plot), {
            responsive: true,
            autosize: true,
            height: 600
        });

        document.getElementById('plotsContainer').scrollIntoView({behavior: 'smooth'});
    } catch (error) {
        console.error('Error generating plots:', error);
        alert('Error generating plots: ' + error.message);
    }
}

function updatePlotLayout(plotId, updates) {
    Plotly.update(plotId, {}, updates);
}

function resetPlotView(plotId) {
    if (plotId === 'surfacePlot') {
        updatePlotLayout(plotId, {
            'scene.camera': { eye: { x: 1.5, y: 1.5, z: 1.3 } }
        });
    } else if (plotId === 'heatmapPlot') {
        Plotly.relayout(plotId, {
            'xaxis.autorange': true,
            'yaxis.autorange': true
        });
    }
}

// Add event listeners for plot customization
document.getElementById('surfacePlot').on('plotly_relayout', function(eventData) {
    // Handle 3D plot view changes
    if (eventData['scene.camera']) {
        updatePlotLayout('surfacePlot', { 'scene.camera': eventData['scene.camera'] });
    }
});

document.getElementById('heatmapPlot').on('plotly_relayout', function(eventData) {
    // Handle heatmap zoom and pan
    if (eventData['xaxis.range[0]'] || eventData['yaxis.range[0]']) {
        updatePlotLayout('heatmapPlot', {
            'xaxis.range': [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']],
            'yaxis.range': [eventData['yaxis.range[0]'], eventData['yaxis.range[1]']]
        });
    }
});
