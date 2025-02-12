import requests
import parsel
import os
import re
import pdfkit
from bs4 import BeautifulSoup

url = 'https://www.cahec.cn/detail/83691.html'
headers = {
    'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0'
}
html_str = '''
<!DOCTYPE html>
    <html lang ="en">
<head>
    <meta charset="UTF-8">
    <title>Document</title >
</ head>
<body>
{article}
</body>
</html>
'''
response = requests.get(url, headers=headers)
response.encoding = 'utf-8'
# print(response.text)

soup = BeautifulSoup(response.text, 'html.parser')

title = soup.select_one('.detail-title').get_text(strip=True)
# print(title)
content = "\n".join([span.get_text(strip=True) for span in soup.select(".current.xcTable p span")])
# print(content)
article = title + '<br>' + content.replace("\n", "<br>")
html = html_str.format(article=article)
html_path = './SpiderHTML/' + title + '.html'
pdf_path = './SpiderPDF/' + title + '.pdf'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
    print(f'{title}保存成功')

config = pdfkit.configuration(wkhtmltopdf=r"E:\wkhtmltopdf\bin\wkhtmltopdf.exe")
pdfkit.from_file(html_path, pdf_path, configuration=config)
