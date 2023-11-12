import glob
import os

from bs4 import BeautifulSoup

def extract_text_from_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = [para.get_text() for para in soup.find_all('span', class_='para')]

    return "\n".join(paragraphs)

if __name__ == '__main__':
    html_directory = 'html'
    output_directory = 'processed'

    # Get all HTML files in the html directory
    file_paths = []
    for path, subdirs, files in os.walk(html_directory):
        for name in files:
            file_paths.append(os.path.join(path, name))

    for f in file_paths:
        output_file_path = os.path.join(output_directory, f[5:-5] + '.txt')

        # Create directory if it does not exist
        directory = os.path.dirname(output_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        out = open(output_file_path, "w")
        out.write(extract_text_from_html_file(f))
        out.close()