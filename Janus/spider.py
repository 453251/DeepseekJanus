import requests
import parsel
import os
import re
import pdfkit

url = 'https://www.bq06.cc/html/46332/1.html'
headers = {
    'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0',
    'Referer':
        'https://link.csdn.net/?from_id=144757254&target=https%3A%2F%2Fwww.bq05.cc%2Fhtml%2F46332%2F1.html',
    'Cookie':
        'Hm_lvt_9c5f07b6ce20e3782eac91ed47d1421c=1738312201; Hm_lpvt_9c5f07b6ce20e3782eac91ed47d1421c=1738312201; HMACCOUNT=DA8D2F966BE0BA45'
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

selector = parsel.Selector(response.text)
# 获取章节名称，注意这里getall返回的是一个列表，列表中含一个元素，所以用[0]索引一下
# ::text表示只获取文本，标签不获取
title = selector.css('#read > div.book.reader > div.content > h1::text').getall()[0]
# print(title)
content = selector.css('#chaptercontent::text').getall()
# 最后三行不是文章内容，所以用content[:-4:]，表示逐个切片，直到倒数第四个元素
content = '\n'.join([p.strip() for p in content[:-4:] if p.strip()])
# print(content)
article = title + '<br>' + content
html = html_str.format(article=article)
html_path = './SpiderHTML/' + title + '.html'
pdf_path = './SpiderPDF/' + title + '.pdf'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
    print(f'{title}保存成功')

config = pdfkit.configuration(wkhtmltopdf=r"E:\wkhtmltopdf\bin\wkhtmltopdf.exe")
pdfkit.from_file(html_path, pdf_path, configuration=config)
