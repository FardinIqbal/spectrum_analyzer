# Spectrum Analyzer

**JWST Spectral Visualization Tool**
A web app for uploading, parsing, and visualizing spectral data from James Webb Space Telescope (JWST) `.fits` files. This tool enables interactive 3D exploration of exoplanet flux data, ideal for researchers, students, and science communicators.

---

## Features

* Upload JWST `.fits` files (`flux`, `wavelength`, `time`)
* Upload complete MAST archive `.zip` directories
* Select custom wavelength bands for targeted spectral analysis
* Generate interactive 3D surface plots and heatmaps using Plotly.js
* Preloads example data on first visit
* Automatically downloads visualizations as standalone HTML files
* Fully customizable color scales and viewing options

---

## Tech Stack

* **Backend:** Python, Flask, Astropy
* **Frontend:** HTML, Tailwind CSS, Plotly.js, JavaScript
* **Deployment:** Heroku
* **Data Format:** NASA FITS (Flexible Image Transport System)

---

## Demo

*"Visualize the unseen."*
![image](https://github.com/user-attachments/assets/9b7d82e4-a99a-48c7-b9e4-6b63db7da11e)

---

## Project Structure

```
spectrum_analyzer/
├── app.py                 # Flask backend logic
├── templates/
│   └── index.html         # Main HTML layout
├── static/
│   ├── css/styles.css     # Tailwind + custom styles
│   └── js/main.js         # All frontend logic
├── plots/                 # Auto-downloaded HTML visualizations
├── preloaded_data/        # Demo FITS files for preload
├── requirements.txt       # Python dependencies
├── Procfile, runtime.txt  # Heroku deployment config
└── README.md
```

---

## Local Setup

1. Clone the repository:

   ```
    git clone https://github.com/your-username/spectrum_analyzer.git
    cd spectrum_analyzer
   ```

2. Create and activate a virtual environment:

   ```
    python -m venv .venv
    source .venv/bin/activate       # macOS/Linux
    .venv\Scripts\activate          # Windows
   ```

3. Install dependencies:

   ```
    pip install -r requirements.txt
   ```

4. Run the development server:

   ```
    python app.py
   ```

5. Visit `http://localhost:5000` in your browser.

---

## Heroku Deployment

1. Log in to Heroku:

   ```
    heroku login
   ```

2. Connect your local repo to a Heroku app:

   ```
    heroku git:remote -a your-heroku-app-name
   ```

3. Deploy the app:

   ```
    git push heroku main
   ```

Ensure your root directory includes a valid `Procfile` and `runtime.txt`.

---

## FITS File Requirements

These files are required for successful visualization:

```
flux.fits        - 3D data cube (intensity over wavelength and time)
wavelength.fits  - 1D or 2D wavelength array
time.fits        - 1D time array matching the flux cube dimensions
```

---

## Example Wavelength Bands

Preloaded wavelength ranges include:

* CH₄ Band: 2.14 – 2.50 μm
* CO Band: 4.50 – 5.05 μm

You may also define your own custom bands through the interface.

---

## Credits

* Built with [Astropy](https://www.astropy.org/) for FITS file parsing
* Visualized using [Plotly.js](https://plotly.com/javascript/)
* JWST data courtesy of NASA’s [MAST Archive](https://mast.stsci.edu/)

---

## License

MIT License – Free to use, modify, and distribute with attribution.

