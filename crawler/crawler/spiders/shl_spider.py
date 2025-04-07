import scrapy

class SHLCatalogSpider(scrapy.Spider):
    name = "shl_catalog"
    start_urls = ["https://www.shl.com/solutions/products/product-catalog/"]
    custom_settings = {
        'CLOSESPIDER_ITEMCOUNT': 500,  # adjust as needed
        'DOWNLOAD_DELAY': 1,            # be polite to the server
        
    }

    def parse(self, response):
        # Extract product links from listing page (both categories)
        links = response.css('a.custom__table-heading__title-link::attr(href)').getall()
        links += response.css('td.custom__table-heading__title a::attr(href)').getall()
        for href in set(links):
            yield response.follow(href, callback=self.parse_detail)

        # Follow "Next" pagination link
        next_page = response.css('li.-next a::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
        # Also follow numbered pagination links to ensure coverage
        for page in response.css('ul.pagination li a::attr(href)').getall():
            yield response.follow(page, callback=self.parse)

    def parse_detail(self, response):
        # Title and URL
        title = response.css('div.content__container h1::text').get(default='').strip()
        url = response.url

        # Description
        description = response.xpath(
            '//div[h4[text()="Description"]]/p/text()'
        ).get(default='No description available').strip()

        # Job levels and languages
        job_levels = response.xpath(
            '//div[h4[text()="Job levels"]]/p/text()'
        ).get(default='Not available').strip()
        languages = response.xpath(
            '//div[h4[text()="Languages"]]/p/text()'
        ).get(default='Not available').strip()

        # Assessment length
        duration = response.xpath(
            '//div[h4[text()="Assessment length"]]/p/text()'
        ).get(default='').strip()

        # Test types (C, P, A, B codes)
        test_type_spans = response.xpath(
            '//p[contains(@class, "product-catalogue__small-text")][contains(., "Test Type")]/span//span[@class="product-catalogue__key"]/text()'
        ).getall()
        test_types = [t.strip() for t in test_type_spans if t.strip()]

        # Remote testing support
        remote = 'Yes' if response.xpath(
            '//p[contains(., "Remote Testing")]/span[contains(@class, "-yes")]'
        ) else 'No'

        # Adaptive/IRT support
        adaptive_raw = response.xpath(
            '//li[contains(., "Adaptive") or contains(., "IRT")]/text()'
        ).get()
        adaptive_supported = (
            adaptive_raw.split(':')[-1].strip() if adaptive_raw and ':' in adaptive_raw else 'No Information'
        )

        yield {
            'title': title,
            'url': url,
            'description': description,
            'job_levels': job_levels,
            'languages': languages,
            'duration_minutes': duration,
            'test_types': test_types,
            'remote_testing': remote,
            'adaptive_supported': adaptive_supported,
        }

    