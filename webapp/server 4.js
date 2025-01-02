const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

// servir les fichiers statiques (comme indez.html) depuis le réportoire courant
app.use(express.static(path.join(__dirname)));

// route pour l'URL principal
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// démarrer le serveur 
app.listen(port, () => {
    console.log(`Serveur lancé sur http://localhost:${port}`);
});