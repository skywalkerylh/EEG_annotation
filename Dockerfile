# FROM python:3.10

# ENV APP_HOME /app
# WORKDIR $APP_HOME
# COPY . ./

# RUN pip install -r requirements.txt

# EXPOSE 8050

# CMD python app.py



# Use an official Python image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY app.py /app/

COPY requirements.txt /app/
# Install any necessary Python packages
RUN pip install -r requirements.txt

EXPOSE 8080

# Specify the command to run your Python application
CMD ["python", "app.py"]
