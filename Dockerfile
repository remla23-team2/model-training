# Use the Python 3.9 image
FROM python:3.9

# Set the working directory
WORKDIR /root

# Copy the requirements file to the container
COPY requirements.txt /root/

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the application code to the container
COPY main.py /root/
COPY /data/ /root/data/

# Set the entrypoint and default command for the container
ENTRYPOINT ["python"]
CMD ["main.py"]
