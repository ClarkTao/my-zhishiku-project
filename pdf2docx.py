from pdf2docx import Converter

def convert_pdf_to_word(pdf_path, docx_path):
    """
    将PDF文件转换为Word文档
    参数:
        pdf_path (str): 输入的PDF文件路径
        docx_path (str): 输出的Word文档路径
    """
    try:
        # 创建一个转换器对象
        cv = Converter(pdf_path)
        # 执行转换，start=0, end=None 表示转换所有页面
        cv.convert(docx_path, start=0, end=None)
        # 关闭转换器释放资源
        cv.close()
        print(f"转换成功！文件已保存至: {docx_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

# 使用示例
if __name__ == "__main__":
    input_pdf = "input.pdf"   # 请替换为你的PDF文件路径
    output_docx = "output.docx" # 请指定输出的Word文件路径
    convert_pdf_to_word(input_pdf, output_docx)