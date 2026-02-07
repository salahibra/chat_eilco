const http = require('http');
const fs = require('fs');
const path = require('path');

const API_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8072';

const server = http.createServer((req, res) => {
  // Proxy API requests
  if (req.url.startsWith('/api')) {
    const backendPath = req.url.replace('/api', '');
    const backendUrl = new URL(backendPath, API_URL);
    
    const options = {
      hostname: backendUrl.hostname,
      port: backendUrl.port,
      path: backendUrl.pathname + backendUrl.search,
      method: req.method,
      headers: req.headers
    };
    
    console.log('Proxying to:', `${options.hostname}:${options.port}${options.path}`);
    
    const proxyReq = http.request(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res);
    });
    
    proxyReq.on('error', (err) => {
      console.error('Proxy error:', err);
      res.writeHead(502);
      res.end('Bad Gateway: ' + err.message);
    });
    
    req.pipe(proxyReq);
  } else {
    // Serve static files
    let filePath = path.join(__dirname, 'dist', req.url);
    
    if (req.url === '/' || !path.extname(filePath)) {
      filePath = path.join(__dirname, 'dist', 'index.html');
    }
    
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end('Not Found');
      } else {
        const ext = path.extname(filePath);
        const mimeTypes = {
          '.html': 'text/html',
          '.js': 'application/javascript',
          '.css': 'text/css',
          '.json': 'application/json',
          '.png': 'image/png',
          '.jpg': 'image/jpeg',
          '.gif': 'image/gif',
          '.svg': 'image/svg+xml'
        };
        res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'text/plain' });
        res.end(data);
      }
    });
  }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Backend API URL: ${API_URL}`);
});
