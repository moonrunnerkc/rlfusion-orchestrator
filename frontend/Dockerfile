# Author: Bradley R. Kinnard
# frontend/Dockerfile - Vite React dev server for RLFusion

FROM node:20-alpine

WORKDIR /app

# Lockfile-aware install for reproducible builds.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

# tini reaps the Vite child processes when the container stops.
RUN apk add --no-cache tini
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["npm", "run", "dev", "--", "--host"]
