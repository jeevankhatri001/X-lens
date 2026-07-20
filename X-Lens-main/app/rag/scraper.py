import httpx
from app.rag.cleaner import clean_html
async def scrape_url(url:str)->str:
    async with httpx.AsyncClient(follow_redirects=True,timeout=30) as client:
        response=await client.get(url,headers={"User-Agent":"X-Lens-Research/0.1"}); response.raise_for_status()
    return clean_html(response.text)
