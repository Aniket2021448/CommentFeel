# CommentFeel
live demo: https://huggingface.co/spaces/GoodML/Comment-Feel

# YouTube Comments Sentiment Analysis

## Description
This project analyzes the sentiment of comments from a YouTube video using a pre-trained sentiment analysis model based on BERT and TensorFlow. It allows users to input a YouTube video URL, fetch comments related to that video, and analyze their sentiments (positive, negative, or neutral).

## Setup Instructions
1. Clone this repository to your local machine.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Replace the placeholder `DEVELOPER_KEY` with your actual YouTube Data API key in the `DEVELOPER_KEY` variable in `CommentFeel.py`.
4. Run the Streamlit app using `streamlit run CommentFeel.py`.

## File Structure
- `CommentFeel.py`: Main Streamlit application file.
- `requirements.txt`: List of required Python libraries and versions.
- `README.md`: Markdown file containing project information (this file).

## Usage
1. Run the Streamlit app as per the setup instructions.
2. Input a valid YouTube video URL in the provided text box.
3. Click on the "Extract Comments and Analyze" button to fetch comments and analyze their sentiment.
4. View the sentiment analysis results in the form of pie and bar charts.

## Dependencies
- Streamlit
- Transformers
- Torch
- Pandas
- Google API Python Client
- Plotly
- NumPy


## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.

## Credits
- Author: Aniket Panchal
- Email: AniketPanchal1257@gmail.com
