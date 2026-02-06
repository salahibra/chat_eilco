# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* yarn.lock* ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM node:18-alpine

WORKDIR /app

# Copy built app from build stage
COPY --from=build /app/dist ./dist

# Copy server script
COPY server.js ./server.js

# Expose port
EXPOSE 3000

# Start the proxy server
CMD ["node", "server.js"]