import win32com.client
import os
import sys


def word_to_pdf(docx_path, pdf_path):
    # 确保路径是绝对路径
    docx_path = os.path.abspath(docx_path)
    pdf_path = os.path.abspath(pdf_path)

    # 检查文件是否存在
    if not os.path.exists(docx_path):
        print(f"错误：文件不存在 - {docx_path}")
        return

    print(f"正在转换: {docx_path} -> {pdf_path}")

    try:
        # 启动Word应用
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False  # 后台运行

        try:
            # 尝试打开文档
            doc = word.Documents.Open(docx_path)

            # 确保输出目录存在
            output_dir = os.path.dirname(pdf_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 另存为PDF
            doc.SaveAs(pdf_path, FileFormat=17)
            print("转换成功！")

        except Exception as e:
            print(f"转换过程中出错: {e}")
            # 获取更详细的错误信息
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"错误类型: {exc_type}")
            print(f"错误位置: {exc_tb.tb_lineno}")

        finally:
            # 确保关闭文档和Word应用
            if 'doc' in locals() and doc is not None:
                doc.Close(SaveChanges=False)
            word.Quit()

    except Exception as e:
        print(f"启动Word应用时出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 使用原始字符串处理路径中的反斜杠
    input_path = r"D:\PYTHON\zhishiku\temp_工程征地移民智慧管理平台-操作手册-chy0311.docx"
    output_path = r"D:\PYTHON\zhishiku\输出的PDF.pdf"

    word_to_pdf(input_path, output_path)