G-Crawler, an auto crawler tool set that designed to let crawling wiki pedia like web site simpler powered by LLM and Scrapy.

> [!CAUTION]
>
> The analyse module still in heavy development, it may reach a dead loop (with a limit constraint), use it by your own risk.



There are 4 executable commands in `g_crawler/cmd.py`:

1. Sample, sample some categories from bwiki, which will be used analyse page structure. A config file is generated in `configs/{site_name}` folder.
2. Analyse, analyse page structure (using LLM) to generate trim rules. The trim rules are a set of xpath selector, the selected elements will be removed when converting html to markdown. Trim Rules will be attached in config file.
3. Test, using generated trim rules to crawl sampled web page to test if the result favorable or not. Crawled markdown file will be saved in `outputs` folder.
4. Crawl, based on trim rule to crawl pages and save it to Aliyun Oss2.



Environment Variables:

* OPENAI_API_KEY
* OPENAI_API_BASE
* G_CRAWLER_EXTRACT_XPATH_LLM, llm model name, used to write xpath.
* G_CRAWLER_TEST_XPATH_LLM, llm model name, used to test xpath.
* AWS_ENDPOINT, the endpoint of Aliyun OSS
* AWS_ACCESS_KEY_ID, the access key ID of Aliyun OSS
* AWS_SECRET_ACCESS_KEY, the secret access key of Aliyun OSS
* DOCUMENTS_BUCKET, the bucket name of Aliyun OSS which save crawled results.





