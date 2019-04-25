# ------ display logs ----------
# docker logs $(container_id)

# --- remove images all ----- --
# docker rmi $(docker images -q)

# --- remove container all ----
# docker rm $(docker ps -a -q) 



docker build -t detector_project .
docker run -d -p 8080:80 detector_project

