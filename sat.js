import express, { json } from 'express';
import { promises as fs } from 'fs';  // Use promises-based fs
import cors from 'cors';  // Import cors
import axios from 'axios';  // Import axios

const app = express();

// Enable CORS for all routes
app.use(cors());
app.use(json());

app.post('/store-data', async (req, res) => {
    const parameters = req.body;
    console.log('Received parameters:', parameters);
    
    const csvLine = `${parameters.mouseMovements.length || 0},${parameters.screenWidth || 0},${parameters.screenHeight || 0},${parameters.browserName || ''},${parameters.userAgent || ''},${parameters.language || ''},${parameters.timeOnPage || 0},${parameters.clicks || 0},${parameters.keyPresses || 0},${parameters.referrer || ''}\n`;
    console.log('CSV Line:', csvLine);

    try {
        await fs.appendFile('data.csv', csvLine);
        console.log('Data appended to CSV file');
    } catch (err) {
        console.error('Error writing to CSV file:', err);
        res.status(500).send('Error saving data');
        return;
    }

    try {
        const response = await axios.post('http://localhost:5000/predict', parameters);
        const prediction = response.data.prediction;

        res.status(200).json({
            message: 'Data saved successfully',
            prediction: prediction === 0 ? 'bot' : 'human'
        });
    } catch (error) {
        console.error('Error getting prediction from Python model:', error);
        res.status(500).send('Error predicting data');
    }
});

// Start the server
app.listen(3000, () => {
    console.log('Node.js server started on port 3000');
});
