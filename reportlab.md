# reportlab

这个包用于导出pdf，更加了灵活，[官方文档](https://docs.reportlab.com/)。

reportlab.platypus (Page Layout and Typography Using Scripts)：
包含以下几种：
Platpus：从上到下，可以被看成具备多个层次
DocTemplates：文档最外层的容器
PageTemplates：各种页面布局的规格
Frames：包含流动的文本和图形的文档区域规范
Flowables：能够被流入文档的文本、图形和段落

![1715154178074](image/reportlab/1715154178074.png)

## 初始化文档


```python
from reportlab.platypus import SimpleDocTemplate

pdf_name = "test.pdf"
pdf = SimpleDocTemplate(pdf_name)

content = []
content.append()

pdf.build(content)
```
