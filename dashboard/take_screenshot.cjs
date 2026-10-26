const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

async function capture() {
  const screenshotDir = path.resolve(__dirname, '../docs/screenshots');
  if (!fs.existsSync(screenshotDir)) {
    fs.mkdirSync(screenshotDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1600,1200'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1200 });

  console.log('Navigating to http://localhost:3000/overview...');
  await page.goto('http://localhost:3000/overview', { waitUntil: 'networkidle0', timeout: 15000 });
  await new Promise((r) => setTimeout(r, 2000));

  const overviewPath = path.join(screenshotDir, 'overview.png');
  await page.screenshot({ path: overviewPath, fullPage: false });
  console.log('Saved overview screenshot to', overviewPath);

  // Segmentation page
  console.log('Navigating to http://localhost:3000/segmentation...');
  await page.goto('http://localhost:3000/segmentation', { waitUntil: 'networkidle0', timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1000));
  const segPath = path.join(screenshotDir, 'segmentation.png');
  await page.screenshot({ path: segPath, fullPage: false });
  console.log('Saved segmentation screenshot to', segPath);

  // Privacy page
  console.log('Navigating to http://localhost:3000/privacy...');
  await page.goto('http://localhost:3000/privacy', { waitUntil: 'networkidle0', timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1000));
  const privacyPath = path.join(screenshotDir, 'privacy.png');
  await page.screenshot({ path: privacyPath, fullPage: false });
  console.log('Saved privacy screenshot to', privacyPath);

  await browser.close();
  console.log('Capture complete!');
}

capture().catch((err) => {
  console.error('Screenshot capture failed:', err);
  process.exit(1);
});
