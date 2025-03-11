FROM node:18-alpine as build
WORKDIR /app
COPY package.json ./

COPY package-lock.json* ./
RUN npm install
COPY . ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

RUN mkdir -p /etc/nginx/ssl

EXPOSE 443
CMD ["nginx", "-g", "daemon off;"]
