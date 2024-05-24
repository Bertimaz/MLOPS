GNU nano 7.2                                                                                        Dockerfile                                                                                                 
FROM orgoro/dlib-opencv-python:latest


# Set the working directory to /app
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the repository into the working directory
RUN git clone -b main https://github.com/Bertimaz/MLOPS.git .

# Optionally, install any dependencies required by your application
RUN pip install -r requirements.txt

#EXPOSE port 5050
EXPOSE 5050

# Set the command to run your application
#CMD ["python","./backend.py"]
