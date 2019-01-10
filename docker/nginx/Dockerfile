FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf

COPY ./docker/nginx/pred_nginx.conf /etc/nginx/conf.d/

EXPOSE 81