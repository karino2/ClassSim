# ClassSim
Repository for research


### Data Scraping

For scraping urlists, we use selenium with headless chrome.
This is only necessary for url retrieval (data setup).

Use Dockerfile.selenium for scraping image and run docker with "--cap-add=SYS_ADMIN" option for headless chrome.