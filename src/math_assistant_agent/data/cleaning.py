import re

from bs4 import BeautifulSoup


def clean_html_for_math(html_text):
    """
    Remove tags HTML, converte parágrafos em quebras de linha e
    preserva o conteúdo das fórmulas (MathJax/LaTeX).
    """
    if not html_text:
        return ""

    soup = BeautifulSoup(html_text, "html.parser")

    # Adiciona quebras de linha duplas após parágrafos, blocos de código e títulos
    for tag in soup.find_all(["p", "br", "h1", "h2", "h3", "pre", "div"]):
        tag.insert_after("\n\n")

    # Transforma tags de lista em bullet points
    for li in soup.find_all("li"):
        li.insert_before("- ")
        li.insert_after("\n")

    # Extrai todo o texto (O BeautifulSoup inteligentemente extrai o conteúdo
    # $...$ de dentro da tag <span class="math-container">)
    texto_limpo = soup.get_text()

    # Remove quebras de linha excessivas (mais de 2 vira apenas 2)
    texto_limpo = re.sub(r"\n{3,}", "\n\n", texto_limpo)

    return texto_limpo.strip()
