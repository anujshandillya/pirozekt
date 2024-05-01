# DocBot

### Description

"DocBot" suggests a platform where users can upload PDF documents and ask questions related to the content of those documents. It implies a central hub or repository for querying information from PDF files. Users can search for specific information, extract data, or seek answers to their queries by interacting with the uploaded PDF documents.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

# Getting Started

## Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/anujshandillya/pirozekt.git
   ```
2. **Create a local env**

   ### Windows

   Go to project directory.

   ```sh
   cd path\to\your\project
   ```

   Create virtual environment.

   ```sh
   python -m venv myenv
   ```

   Activate the environment.

   ```sh
   myenv\Scripts\activate
   ```

   ### Linux/MacOS

   Go to project directory.

   ```sh
   cd path\to\your\project
   ```

   Create virtual environment.

   ```sh
   python -m venv myenv
   ```

   Activate the environment.

   ```sh
   source myenv/bin/activate
   ```

3. **Get your Cohere API key**

   - Go to <a href="http://www.cohere.com/">Cohere</a> and get your own API key.

4. **Create a .env file and load it**
   Create a file .env

   ```bash
   touch .env
   ```

   Paste your API key

   ```
   API_KEY='<your_api_key>'
   BASE="Based on the passage above, answer the following question and if it is not mentioned then just return 'Sorry, there is nothing related to this topic in the given text'"
   ```

   Load it

   ```py
   from dotenv import load_dotenv
   load_dotenv()
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Start the application by:
  ```bash
  streamlit run app.py
  ```
- Select input type: **PDF**, **TEXT**.
- Ask questions related to the uploaded document.
- It will give answer related to the query.

### Libraries used:

1. **cohere**

   - This module is imported, presumably for specific functionality provided by the 'cohere' library.

2. **NumPy (np)**

   - NumPy is imported and aliased as 'np'. It is widely used for numerical computing, providing powerful array operations and mathematical functions.

3. **norm**

   - The 'norm' function is imported from the 'numpy.linalg' module. This function calculates the norm (magnitude) of a vector or matrix.

4. **Pandas (pd)**

   - Pandas is imported and aliased as 'pd'. It is used for data manipulation and analysis, offering data structures like DataFrame and tools for reading and writing data.

5. **pdfplumber**

   - This module is imported, likely used for extracting data from PDF documents.

6. **Streamlit (st)**

   - Streamlit is imported and aliased as 'st'. It is a popular framework for building interactive web applications for data science and machine learning.

7. **load_dotenv (from dotenv)**

   - The 'load_dotenv' function is imported from the 'dotenv' module. This function is used to load environment variables from a .env file into the environment.

8. **os**
   - The 'os' module is imported, providing operating system-dependent functionality, such as interacting with the file system and environment variables.

## Contributing

This project welcomes contributions and suggestions.

## License

[License](LICENSE)

