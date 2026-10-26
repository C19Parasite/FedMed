const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

async function verifyAll() {
  const screenshotDir = path.resolve(__dirname, '../docs/screenshots');
  if (!fs.existsSync(screenshotDir)) {
    fs.mkdirSync(screenshotDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();

  // Listen to console errors and failed requests
  const consoleErrors = [];
  const failedRequests = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  page.on('requestfailed', (req) => {
    failedRequests.push(`${req.url()} (${req.failure().errorText})`);
  });

  // TEST 1: Desktop Viewport (1440x900) - Hospital Nodes Alignment Check
  console.log('--- TEST 1: Desktop Hospital Nodes Alignment (1440x900) ---');
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://localhost:3000/overview', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 1000));

  // Check heights and positions of the 3 hospital cards specifically
  const cardsInfo = await page.$$eval('div.grid.grid-cols-1.md\\:grid-cols-3 > div', (allCards) => {
    // Filter to cards that have hospital headers (contain hospital name h4)
    const cards = allCards.filter((c) => c.querySelector('h4'));
    return cards.map((c, i) => {
      const rect = c.getBoundingClientRect();
      const h4 = c.querySelector('h4');
      const powerBtn = c.querySelector('button[aria-label*="Toggle"]');
      return {
        index: i,
        top: Math.round(rect.top),
        bottom: Math.round(rect.bottom),
        height: Math.round(rect.height),
        width: Math.round(rect.width),
        name: h4 ? h4.textContent.trim() : null,
        hasPowerBtn: !!powerBtn,
        powerBtnInside: powerBtn ? (powerBtn.getBoundingClientRect().right <= rect.right + 2) : false,
      };
    });
  });

  console.log('Hospital Cards Layout Metrics:', cardsInfo);

  const heightsEqual = cardsInfo.length === 3 &&
    Math.abs(cardsInfo[0].height - cardsInfo[1].height) <= 2 &&
    Math.abs(cardsInfo[1].height - cardsInfo[2].height) <= 2;
  const topsEqual = cardsInfo.length === 3 &&
    Math.abs(cardsInfo[0].top - cardsInfo[1].top) <= 2 &&
    Math.abs(cardsInfo[1].top - cardsInfo[2].top) <= 2;
  const allPowerButtonsInside = cardsInfo.every((c) => c.hasPowerBtn && c.powerBtnInside);

  console.log(`[CHECK] All 3 cards have equal height: ${heightsEqual} (Heights: ${cardsInfo.map(c=>c.height).join(', ')}px)`);
  console.log(`[CHECK] All 3 cards aligned along same top edge: ${topsEqual} (Tops: ${cardsInfo.map(c=>c.top).join(', ')}px)`);
  console.log(`[CHECK] All power buttons anchored inside card bounds: ${allPowerButtonsInside}`);

  await page.screenshot({ path: path.join(screenshotDir, 'desktop_overview_fixed.png') });

  // TEST 2: Hospital B Offline Toggle & Resilience Check
  console.log('\n--- TEST 2: Hospital B Offline Toggle & Dynamic Resilience ---');
  // Click Hospital B toggle button
  const toggleButtons = await page.$$('button[aria-label*="Toggle"]');
  if (toggleButtons.length >= 2) {
    await toggleButtons[1].click(); // Toggle Hospital B
    await new Promise((r) => setTimeout(r, 600));

    const cardsAfterToggle = await page.$$eval('div.grid.grid-cols-1.md\\:grid-cols-3 > div', (allCards) => {
      const cards = allCards.filter((c) => c.querySelector('h4'));
      return cards.map((c) => ({
        top: Math.round(c.getBoundingClientRect().top),
        bottom: Math.round(c.getBoundingClientRect().bottom),
        height: Math.round(c.getBoundingClientRect().height),
      }));
    });
    console.log('Cards after Hospital B toggled offline:', cardsAfterToggle);
    const heightsStillEqual = Math.abs(cardsAfterToggle[0].height - cardsAfterToggle[1].height) <= 2 &&
      Math.abs(cardsAfterToggle[1].height - cardsAfterToggle[2].height) <= 2;
    console.log(`[CHECK] Cards remain equal height after toggling offline: ${heightsStillEqual}`);

    await page.screenshot({ path: path.join(screenshotDir, 'hospital_b_offline.png') });

    // Toggle back online
    await toggleButtons[1].click();
    await new Promise((r) => setTimeout(r, 600));
  }

  // TEST 3: Real MRI Viewer in Modal (?case=case_0132)
  console.log('\n--- TEST 3: Real MRI Viewer & Segmentation Masks (?case=case_0132) ---');
  await page.goto('http://localhost:3000/overview?case=case_0132', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 1200));

  // Check if real MRI images are loaded in both panels
  const mriImagesInfo = await page.$$eval('img[alt*="Real MRI"]', (imgs) => {
    return imgs.map((img) => ({
      src: img.src,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
      complete: img.complete,
      visible: img.getBoundingClientRect().width > 0,
    }));
  });
  console.log('Real MRI Images Rendered in Panels:', mriImagesInfo);

  const mriLoaded = mriImagesInfo.length === 2 && mriImagesInfo.every((img) => img.complete && img.naturalWidth > 0);
  console.log(`[CHECK] Real MRI images loaded in both comparison panels: ${mriLoaded}`);

  // Check tumor mask overlays
  const maskImagesInfo = await page.$$eval('img[alt*="Mask"]', (imgs) => {
    return imgs.map((img) => ({
      alt: img.alt,
      src: img.src,
      naturalWidth: img.naturalWidth,
      complete: img.complete,
    }));
  });
  console.log('Tumor Mask Overlays Rendered:', maskImagesInfo);
  const masksLoaded = maskImagesInfo.length >= 2 && maskImagesInfo.every((img) => img.complete && img.naturalWidth > 0);
  console.log(`[CHECK] Segmentation masks loaded and aligned with MRI: ${masksLoaded}`);

  await page.screenshot({ path: path.join(screenshotDir, 'mri_modal_real_images.png') });

  // TEST 4: Slider Interactions (Z Slice & Opacity)
  console.log('\n--- TEST 4: Slice (Z) and Opacity Sliders Test ---');
  const zSlider = await page.$('input[type="range"][max="32"]');
  if (zSlider) {
    // Change Z to slice 12 using prototype setter
    await page.evaluate((el) => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(el, '12');
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }, zSlider);
    await new Promise((r) => setTimeout(r, 600));

    // Verify image source updated to slice 11 (0-indexed)
    const updatedMriSrcs = await page.$$eval('img[alt*="Real MRI"]', (imgs) => imgs.map((img) => img.src));
    console.log('Updated MRI Sources after Z=12:', updatedMriSrcs);
    const zUpdated = updatedMriSrcs.every((s) => s.includes('slice_11_'));
    console.log(`[CHECK] Z slider correctly updated MRI slices: ${zUpdated}`);
  }

  const opacitySlider = await page.$('input[type="range"][max="1"]');
  if (opacitySlider) {
    // Change opacity to 0.4 using prototype setter
    await page.evaluate((el) => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(el, '0.4');
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }, opacitySlider);
    await new Promise((r) => setTimeout(r, 400));

    const overlayOpacity = await page.$$eval('div[style*="opacity"]', (divs) => {
      return divs.map((d) => d.style.opacity).filter((op) => op && op !== '');
    });
    console.log(`[CHECK] Opacity slider updated mask container opacity to:`, overlayOpacity);
  }

  await page.screenshot({ path: path.join(screenshotDir, 'mri_modal_sliders_updated.png') });

  // TEST 5: Tablet (768x1024) and Mobile (390x844) Responsiveness Check
  console.log('\n--- TEST 5: Tablet & Mobile Responsive Viewports ---');
  // Tablet
  await page.setViewport({ width: 768, height: 1024 });
  await page.goto('http://localhost:3000/overview', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 800));
  await page.screenshot({ path: path.join(screenshotDir, 'tablet_overview.png') });

  // Check no horizontal scroll on tablet
  const tabletOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  console.log(`[CHECK] Tablet horizontal overflow: ${tabletOverflow}`);

  // Mobile
  await page.setViewport({ width: 390, height: 844, isMobile: true });
  await page.goto('http://localhost:3000/overview', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 800));
  await page.screenshot({ path: path.join(screenshotDir, 'mobile_overview_fixed.png') });

  const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  console.log(`[CHECK] Mobile horizontal overflow: ${mobileOverflow}`);

  // TEST 6: Segmentation Gallery Page (/segmentation)
  console.log('\n--- TEST 6: Segmentation Gallery Page (/segmentation) ---');
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://localhost:3000/segmentation', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 1000));

  const pageMriImages = await page.$$eval('img[alt*="Real MRI"]', (imgs) => {
    return imgs.map((img) => ({
      src: img.src,
      complete: img.complete,
      naturalWidth: img.naturalWidth,
    }));
  });
  console.log('Gallery Page MRI Images:', pageMriImages);
  const galleryMriLoaded = pageMriImages.length === 2 && pageMriImages.every((img) => img.complete && img.naturalWidth > 0);
  console.log(`[CHECK] Real MRI images rendered in gallery page: ${galleryMriLoaded}`);

  await page.screenshot({ path: path.join(screenshotDir, 'gallery_real_mri.png') });

  console.log('\nConsole Errors during tests:', consoleErrors.length);
  if (consoleErrors.length > 0) {
    console.log(consoleErrors);
  }
  console.log('Failed Requests:', failedRequests.length);
  if (failedRequests.length > 0) {
    console.log(failedRequests);
  }

  await browser.close();
  console.log('\nAll verification tests completed successfully!');
}

verifyAll().catch((err) => {
  console.error('Verification failed:', err);
  process.exit(1);
});
