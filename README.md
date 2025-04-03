<p align="center">
  <img src="frontend/static/logo.png" alt="Data Down Under Logo" width="200">
</p>

# Batch File Uploader and Viewer

## Tool Description

This tool allows you to upload and manage **videos** and **images** through a user-friendly interface. Uploaded files are stored in a **FastAPI** backend, where metadata is extracted and saved. The **Streamlit** frontend provides an intuitive way to view, play, and analyze files.

Key features include:
- **Batch Uploads**: Upload multiple files (videos or images) at once.
- **Metadata Extraction**: Automatically extract and display metadata for each file.
- **File Management**: View, play, and delete uploaded files.
- **Analytics**: Gain insights into your uploaded files (coming soon).

---

## Getting Started

### Start the Application

Run the following command to build and start the containers:

```bash
docker-compose up
```

This will:
1. Build the Docker images for the FastAPI and Streamlit services.
2. Start the containers and serve the application.

---

## Features

### **FastAPI Backend**
- Hosted on: [http://localhost:8000](http://localhost:8000)
- Interactive API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### **Streamlit Frontend**
- Hosted on: [http://localhost:8501](http://localhost:8501)

### **Core Pages**
1. **Dashboard**:  
   View an overview of uploaded files, including total counts of videos and images, and a list of recent uploads.
   
2. **Upload**:  
   Upload multiple videos or images at once. Metadata is extracted automatically during the upload process.
   
3. **View Files**:  
   Browse and interact with uploaded files. Play videos, view images, and explore metadata in a collapsible section.
   
4. **Analytics**:  
   Analyze uploaded files and gain insights (feature coming soon).

---

## Prerequisites

Before starting, ensure the following tools are installed on your system:
- **Docker**
- **Docker Compose**

---

## Access the Application

- **FastAPI Backend**:  
  Visit [http://localhost:8000](http://localhost:8000) to access the API.  
  Documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

- **Streamlit Frontend**:  
  Visit [http://localhost:8501](http://localhost:8501) to interact with the frontend.

---

## Development Workflow

### Live Reloading
Both FastAPI and Streamlit support hot reloading. Any changes made to the code will automatically reflect in the running containers.

### Stopping the Application
To stop the application, press `Ctrl+C` or run the following command:

```bash
docker-compose down
```

This will stop and remove the containers, but the built images will remain.

---

## Directory Structure

The project structure is as follows:

```shell
.
├── backend/                        # FastAPI application
│   ├── classes/                    # Backend helper classes
|   |   |──backend_class_example.py
│   ├── config/                     # Configuration files
|   |   |──settings.py
│   ├── metadata/                   # Metadata storage
│   ├── models/                     # Pydantic schemas
|   |   |──schemas.py
│   ├── routers/                    # API route definitions
|   |   |──files.py
│   ├── services/                   # Business logic services
|   |   |──file_services.py
│   ├── uploaded_files/             # Uploaded files storage
│   ├── uploads/                    # Temporary upload storage
│   ├── main.py                     # FastAPI entrypoint
│   ├── requirements.txt            # Python dependencies for FastAPI
│   └── Dockerfile                  # Dockerfile for FastAPI
├── frontend/                       # Streamlit application
│   ├── classes/                    # Frontend helper classes
|   |   |──frontend_class_example.py
│   ├── components/                 # UI components
|   |   |──layout.py
│   ├── models/                     # Pydantic schemas
|   |   |──schemas.py
│   ├── pages/                      # Streamlit pages
|   |   |──analytics.py
|   |   |──dashboard.py
|   |   |──upload.py
|   |   |──view_files.py
│   ├── routers/                    # Frontend route handlers
|   |   |──file_display.py
|   |   |──file_upload.py
│   ├── static/                     # Static assets (e.g., logo)
|   |   |──logo.py
│   ├── utils/                      # Utility functions
|   |   |──api_client.py
│   ├── app.py                      # Streamlit entrypoint
│   ├── requirements.txt            # Python dependencies for Streamlit
│   └── Dockerfile                  # Dockerfile for Streamlit
├── docker-compose.yml              # Docker Compose configuration
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```

---

## Troubleshooting

- Ensure Docker and Docker Compose are installed and running on your system.
- Verify that the required ports (8000 and 8501) are not in use by other applications.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
