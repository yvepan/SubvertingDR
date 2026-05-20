"""Web scraper module for GPT Researcher.

This module provides the Scraper class that extracts content from URLs
using various scraping backends (BeautifulSoup, PyMuPDF, Browser, etc.).
"""

import asyncio
import importlib
import logging
import subprocess
import sys

import requests
from colorama import Fore, init

from gpt_researcher.utils.workers import WorkerPool

from . import (
    ArxivScraper,
    BeautifulSoupScraper,
    BrowserScraper,
    FireCrawl,
    NoDriverScraper,
    PyMuPDFScraper,
    TavilyExtract,
    WebBaseLoaderScraper,
)


class Scraper:
    """
    Scraper class to extract the content from the links
    """

    def __init__(self, urls, user_agent, scraper, worker_pool: WorkerPool):
        """
        Initialize the Scraper class.
        Args:
            urls:
        """
        self.urls = urls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.scraper = scraper
        if self.scraper == "tavily_extract":
            self._check_pkg(self.scraper)
        if self.scraper == "firecrawl":
            self._check_pkg(self.scraper)
        self.logger = logging.getLogger(__name__)
        self.worker_pool = worker_pool
        self._doc_path = None  # Cache for doc_path

    async def _extract_local_document(self, virtual_url):
        """
        Web poison: extract local document content from a virtual URL.
        Virtual URL format: http://research/{slug}{ext}
        Look up documents from DOC_PATH_WEB_POISON.
        """
        try:
            import os
            import re
            from gpt_researcher.config import Config

            # Get doc_path from config using the web-poison path
            if self._doc_path is None:
                cfg = Config()
                if hasattr(cfg, 'doc_path_web_poison') and cfg.doc_path_web_poison:
                    self._doc_path = cfg.doc_path_web_poison
                    self.logger.info(f"Web poison: using path {self._doc_path}")
                else:
                    self._doc_path = cfg.doc_path

            if not self._doc_path:
                self.logger.error("DOC_PATH_WEB_POISON not configured for web poison documents")
                return {"url": virtual_url, "raw_content": None, "image_urls": [], "title": ""}

            # Extract filename from virtual URL: http://research/my-doc.pdf -> my-doc.pdf
            filename_with_ext = virtual_url.replace("http://research/", "")

            # Find the actual file in doc_path
            actual_file_path = None
            paths_to_search = [self._doc_path] if isinstance(self._doc_path, (str, bytes, os.PathLike)) else self._doc_path

            for path in paths_to_search:
                if os.path.isfile(path):
                    # Check if this file matches
                    base_name, ext = os.path.splitext(os.path.basename(path))
                    slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", base_name).strip("-").lower()
                    if f"{slug}{ext}" == filename_with_ext:
                        actual_file_path = path
                        break
                elif os.path.isdir(path):
                    # Search in directory
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            base_name, ext = os.path.splitext(file)
                            slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", base_name).strip("-").lower()
                            if f"{slug}{ext}" == filename_with_ext:
                                actual_file_path = os.path.join(root, file)
                                break
                        if actual_file_path:
                            break

            if not actual_file_path:
                self.logger.error(f"Local document not found for virtual URL: {virtual_url}")
                return {"url": virtual_url, "raw_content": None, "image_urls": [], "title": ""}

            # Load the document using appropriate loader
            from langchain_community.document_loaders import (
                PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
                UnstructuredPowerPointLoader, UnstructuredCSVLoader,
                UnstructuredExcelLoader, UnstructuredMarkdownLoader, BSHTMLLoader
            )

            ext = os.path.splitext(actual_file_path)[1].lower()
            loader_map = {
                '.pdf': PyMuPDFLoader,
                '.txt': TextLoader,
                '.doc': UnstructuredWordDocumentLoader,
                '.docx': UnstructuredWordDocumentLoader,
                '.pptx': UnstructuredPowerPointLoader,
                '.csv': UnstructuredCSVLoader,
                '.xls': UnstructuredExcelLoader,
                '.xlsx': UnstructuredExcelLoader,
                '.md': UnstructuredMarkdownLoader,
                '.html': BSHTMLLoader,
            }

            loader_class = loader_map.get(ext)
            if not loader_class:
                self.logger.error(f"Unsupported file type: {ext}")
                return {"url": virtual_url, "raw_content": None, "image_urls": [], "title": ""}

            # Load document
            loader = loader_class(actual_file_path)
            pages = await asyncio.to_thread(loader.load)

            # Combine all pages
            content = "\n\n".join([page.page_content for page in pages if page.page_content])
            title = os.path.basename(actual_file_path)

            self.logger.info(f"Web poison: loaded document {title} ({len(content)} chars)")

            return {
                "url": virtual_url,
                "raw_content": content,
                "image_urls": [],
                "title": title,
            }

        except Exception as e:
            self.logger.error(f"Error extracting local document {virtual_url}: {e}")
            return {"url": virtual_url, "raw_content": None, "image_urls": [], "title": ""}

    async def run(self):
        """
        Extracts the content from the links
        """
        contents = await asyncio.gather(
            *(self.extract_data_from_url(url, self.session) for url in self.urls)
        )

        res = [content for content in contents if content["raw_content"] is not None]
        return res

    def _check_pkg(self, scrapper_name: str) -> None:
        """
        Checks and ensures required Python packages are available for scrapers that need
        dependencies beyond requirements.txt. When adding a new scraper to the repo, update `pkg_map`
        with its required information and call check_pkg() during initialization.
        """
        pkg_map = {
            "tavily_extract": {
                "package_installation_name": "tavily-python",
                "import_name": "tavily",
            },
            "firecrawl": {
                "package_installation_name": "firecrawl-py",
                "import_name": "firecrawl",
            },
        }
        pkg = pkg_map[scrapper_name]
        if not importlib.util.find_spec(pkg["import_name"]):
            pkg_inst_name = pkg["package_installation_name"]
            init(autoreset=True)
            print(Fore.YELLOW + f"{pkg_inst_name} not found. Attempting to install...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg_inst_name]
                )
                print(Fore.GREEN + f"{pkg_inst_name} installed successfully.")
            except subprocess.CalledProcessError:
                raise ImportError(
                    Fore.RED
                    + f"Unable to install {pkg_inst_name}. Please install manually with "
                    f"`pip install -U {pkg_inst_name}`"
                )

    async def extract_data_from_url(self, link, session):
        """
        Extracts the data from the link with logging
        """
        async with self.worker_pool.throttle():
            try:
                # Handle virtual research URLs for local documents
                if link.startswith("http://research/"):
                    return await self._extract_local_document(link)

                Scraper = self.get_scraper(link)
                scraper = Scraper(link, session)

                # Get scraper name
                scraper_name = scraper.__class__.__name__
                self.logger.info(f"\n=== Using {scraper_name} ===")

                # Get content
                if hasattr(scraper, "scrape_async"):
                    content, image_urls, title = await scraper.scrape_async()
                else:
                    (
                        content,
                        image_urls,
                        title,
                    ) = await asyncio.get_running_loop().run_in_executor(
                        self.worker_pool.executor, scraper.scrape
                    )

                if len(content) < 100:
                    self.logger.warning(f"Content too short or empty for {link}")
                    return {
                        "url": link,
                        "raw_content": None,
                        "image_urls": [],
                        "title": title,
                    }

                # Log results
                self.logger.info(f"\nTitle: {title}")
                self.logger.info(
                    f"Content length: {len(content) if content else 0} characters"
                )
                self.logger.info(f"Number of images: {len(image_urls)}")
                self.logger.info(f"URL: {link}")
                self.logger.info("=" * 50)

                if not content or len(content) < 100:
                    self.logger.warning(f"Content too short or empty for {link}")
                    return {
                        "url": link,
                        "raw_content": None,
                        "image_urls": [],
                        "title": title,
                    }

                return {
                    "url": link,
                    "raw_content": content,
                    "image_urls": image_urls,
                    "title": title,
                }

            except Exception as e:
                self.logger.error(f"Error processing {link}: {str(e)}")
                return {"url": link, "raw_content": None, "image_urls": [], "title": ""}

    def get_scraper(self, link):
        """
        The function `get_scraper` determines the appropriate scraper class based on the provided link
        or a default scraper if none matches.

        Args:
          link: The `get_scraper` method takes a `link` parameter which is a URL link to a webpage or a
        PDF file. Based on the type of content the link points to, the method determines the appropriate
        scraper class to use for extracting data from that content.

        Returns:
          The `get_scraper` method returns the scraper class based on the provided link. The method
        checks the link to determine the appropriate scraper class to use based on predefined mappings
        in the `SCRAPER_CLASSES` dictionary. If the link ends with ".pdf", it selects the
        `PyMuPDFScraper` class. If the link contains "arxiv.org", it selects the `ArxivScraper
        """

        SCRAPER_CLASSES = {
            "pdf": PyMuPDFScraper,
            "arxiv": ArxivScraper,
            "bs": BeautifulSoupScraper,
            "web_base_loader": WebBaseLoaderScraper,
            "browser": BrowserScraper,
            "nodriver": NoDriverScraper,
            "tavily_extract": TavilyExtract,
            "firecrawl": FireCrawl,
        }

        scraper_key = None

        if link.endswith(".pdf"):
            scraper_key = "pdf"
        elif "arxiv.org" in link:
            scraper_key = "arxiv"
        else:
            scraper_key = self.scraper

        scraper_class = SCRAPER_CLASSES.get(scraper_key)
        if scraper_class is None:
            raise Exception("Scraper not found.")

        return scraper_class
