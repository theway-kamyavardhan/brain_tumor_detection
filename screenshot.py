import asyncio
from playwright.async_api import async_playwright
import time
import os

async def main():
    async with async_playwright() as p:
        # Launch using Brave Browser as requested
        browser = await p.chromium.launch(executable_path=r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe", headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        
        print("Navigating to local site...")
        await page.goto("http://localhost:8502/")
        await page.wait_for_timeout(3000)
        os.makedirs("screenshots", exist_ok=True)
        
        # 1. Whole UI Screen (Home)
        await page.screenshot(path="screenshots/home_ui.png", full_page=True)
        print("Captured Home page")

        # 2. Upload and Predict (Comparisons)
        await page.locator("p:has-text('Upload & Predict')").first.click()
        await page.wait_for_timeout(2000)
        await page.locator('p:has-text("Use Default Test Image")').first.click()
        await page.wait_for_timeout(1000)
        
        await page.locator('button:has-text("Run Analysis")').click()
        print("Waiting for predictions to finish...")
        await page.wait_for_timeout(10000) # give it 10 seconds to process both models
        
        # Screenshot the Model Comparison UI
        await page.screenshot(path="screenshots/model_comparison_ui.png", full_page=True)
        print("Captured Model Comparison UI")
        
        # 3. Grad-CAM UI
        await page.locator("p:has-text('Grad-CAM Visualization')").first.click()
        await page.wait_for_timeout(3000)
        await page.screenshot(path="screenshots/gradcam_ui.png", full_page=True)
        print("Captured Grad-CAM UI")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
