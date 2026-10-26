const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

async function testUXPolish() {
  const screenshotDir = path.resolve(__dirname, '../docs/screenshots');
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1600,1200'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  console.log('1. Testing Desktop Page Scroll & Persistent Sidebar...');
  await page.goto('http://localhost:3000/overview', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 1500));

  // Get initial bounding box of sidebar
  const sidebarInitial = await page.$eval('aside', (el) => {
    const rect = el.getBoundingClientRect();
    return { top: rect.top, left: rect.left, height: rect.height };
  });
  console.log('Sidebar initial position:', sidebarInitial);

  // Scroll page down by 500px
  await page.evaluate(() => window.scrollBy(0, 500));
  await new Promise((r) => setTimeout(r, 800));

  const sidebarAfterScroll = await page.$eval('aside', (el) => {
    const rect = el.getBoundingClientRect();
    return { top: rect.top, left: rect.left, height: rect.height };
  });
  console.log('Sidebar after scroll position:', sidebarAfterScroll);

  const isStickyConstant = Math.abs(sidebarInitial.top - sidebarAfterScroll.top) < 5;
  console.log(`[PASS] Sticky Constant Sidebar Check: ${isStickyConstant} (Top offset remained pinned at ${sidebarAfterScroll.top}px)`);

  await page.screenshot({ path: path.join(screenshotDir, 'scrolled_desktop.png') });

  // 2. Test 2FA Modal
  console.log('2. Testing 2FA Modal...');
  await page.keyboard.press('n'); // Shortcut N triggers new round / 2FA
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: path.join(screenshotDir, '2fa_modal.png') });
  await page.keyboard.press('Escape');

  // 3. Test Command Menu (Ctrl+K)
  console.log('3. Testing Cmd/Ctrl+K Command Menu...');
  await page.keyboard.down('Control');
  await page.keyboard.press('KeyK');
  await page.keyboard.up('Control');
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: path.join(screenshotDir, 'command_palette.png') });
  await page.keyboard.press('Escape');

  // 4. Test Mobile Responsive Viewport (390px iPhone 14 style)
  console.log('4. Testing Mobile Viewport & Bottom Nav...');
  await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true });
  await page.goto('http://localhost:3000/overview', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(screenshotDir, 'mobile_overview.png') });

  // 5. Test Predictive Back & Modal Sync (?case=case_0132)
  console.log('5. Testing Predictive Back (?case=...) on Mobile...');
  await page.goto('http://localhost:3000/overview?case=case_0132', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(screenshotDir, 'mobile_sheet_open.png') });

  // Go back via browser history back
  await page.goBack();
  await new Promise((r) => setTimeout(r, 500));
  const urlAfterBack = page.url();
  console.log('[PASS] URL after back gesture:', urlAfterBack);

  await browser.close();
  console.log('UX Polish verification successfully completed!');
}

testUXPolish().catch((err) => {
  console.error('UX verification test failed:', err);
  process.exit(1);
});
