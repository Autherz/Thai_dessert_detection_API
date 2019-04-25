FROM ubuntu:latest

RUN apt-get update --fix-missing



# Install virtualenv, nginx, supervisor
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install -y  python3.6
RUN apt-get install -y build-essential git
RUN apt-get install -y python python3-dev python3-setuptools
RUN apt-get install -y python3-pip python3-venv
RUN apt-get install -y nginx supervisor
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender1
# RUN apt-get install -y python3.6-tk
RUN service supervisor stop

# create virtual env and install dependencies
# Due to a bug with h5 we install Cython first


RUN python3 -m venv /opt/venv
ADD ./requirements.txt /opt/venv/requirements.txt
RUN /opt/venv/bin/pip3 install Cython && /opt/venv/bin/pip3 install -r /opt/venv/requirements.txt

# expose port
EXPOSE 80

RUN pip3 install supervisor-stdout
RUN pip3 install circus

# Add our config files
ADD ./circus.conf /etc/circus.conf
ADD ./supervisord.conf /etc/supervisord.conf
ADD ./nginx.conf /etc/nginx/nginx.conf

# Copy our service code
ADD ./service /opt/app

# restart nginx to load the config
RUN service nginx stop

# start supervisor to run our wsgi server
CMD ["circusd", "/etc/circus.conf"]
# CMD ["/opt/venv/bin/gunicorn", "main:app"] 