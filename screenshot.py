import asyncio
from playwright.async_api import async_playwright
import time
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        
        # Go to home page
        await page.goto("http://localhost:8502/")
        await page.wait_for_timeout(3000)
        os.makedirs("screenshots", exist_ok=True)
        await page.screenshot(path="screenshots/home.png")
        print("Captured Home page")

        # Click Upload & Predict
        await page.locator("text=Upload & Predict").click()
        await page.wait_for_timeout(2000)
        await page.screenshot(path="screenshots/upload_initial.png")
        print("Captured Upload & Predict (initial)")

        # Click Use Default Test Image
        # st.radio label 'Input Method'
        await page.locator('text=Use Default Test Image').click()
        await page.wait_for_timeout(1000)
        
        # Click Run Analysis button
        await page.locator('button:has-text("Run Analysis")').click()
        # Wait for the processing to finish (spinner goes away, success message appears)
        print("Waiting for predictions to finish...")
        await page.wait_for_timeout(10000) # give it 10 seconds to process both models
        
        # Screenshot the predictions and probability graphs
        await page.screenshot(path="screenshots/predictions.png", full_page=True)
        print("Captured Predictions")
        
        # Go to Architecture & Research tab
        await page.locator("text=Architecture & Research").click()
        await page.wait_for_timeout(3000)
        await page.screenshot(path="screenshots/architecture.png", full_page=True)
        print("Captured Architecture")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
