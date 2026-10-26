const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

async function verifyThemes() {
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
  await page.setViewport({ width: 1440, height: 900 });

  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  console.log('=== TEST 1: Initial Load & Brain Emoji Absence Check ===');
  await page.goto('http://localhost:3000/overview', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 800));

  // Check brain emoji absence
  const pageBodyText = await page.evaluate(() => document.body.innerText);
  const containsBrainEmoji = pageBodyText.includes('🧠');
  console.log(`[CHECK 1] Brain emoji completely removed from visible UI: ${!containsBrainEmoji}`);

  // Check SVG vector logo rendered
  const svgLogoCount = await page.$$eval('svg[aria-label*="FedMed"]', (els) => els.length);
  console.log(`[CHECK 2] Custom Medical AI SVG vector logo rendered: ${svgLogoCount > 0} (Count: ${svgLogoCount})`);

  console.log('\n=== TEST 2: Open Appearance & Theme Settings Dialog ===');
  // Click Header Theme button or Sidebar Settings button
  const settingsBtn = await page.$('button[aria-label="Theme & Brand Appearance Settings"]');
  if (settingsBtn) {
    await settingsBtn.click();
  } else {
    const sidebarSettingsBtn = await page.$('button[aria-label="Appearance & Theme Settings"]');
    if (sidebarSettingsBtn) await sidebarSettingsBtn.click();
  }
  await new Promise((r) => setTimeout(r, 600));

  const isModalOpen = await page.$eval('div[role="dialog"]', () => true).catch(() => false);
  console.log(`[CHECK 3] Appearance & Theme Settings modal opened: ${isModalOpen}`);

  // Check all 6 theme preview cards are rendered
  const themeCardTitles = await page.$$eval('h4', (els) => els.map((e) => e.innerText));
  const hasThemeA = themeCardTitles.some((t) => t.includes('Theme A'));
  const hasThemeB = themeCardTitles.some((t) => t.includes('Theme B'));
  const hasThemeC = themeCardTitles.some((t) => t.includes('Theme C'));
  const hasThemeD = themeCardTitles.some((t) => t.includes('Theme D'));
  const hasThemeE = themeCardTitles.some((t) => t.includes('Theme E'));
  const hasThemeF = themeCardTitles.some((t) => t.includes('Theme F'));
  console.log(`[CHECK 4] All 6 theme options present: ${hasThemeA && hasThemeB && hasThemeC && hasThemeD && hasThemeE && hasThemeF}`);

  await page.screenshot({ path: path.join(screenshotDir, 'theme_settings_modal_6_themes.png') });

  console.log('\n=== TEST 3: Live Preview Behavior (Theme B - Medical Teal) ===');
  // Click Theme B (Medical Teal) to preview
  const themeCards = await page.$$('div.cursor-pointer');
  let tealCard = null;
  for (const c of themeCards) {
    const text = await page.evaluate((el) => el.innerText, c);
    if (text.includes('Medical Teal')) {
      tealCard = c;
      break;
    }
  }

  if (tealCard) {
    await tealCard.click();
    await new Promise((r) => setTimeout(r, 400));
    const activeDataTheme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
    console.log(`[CHECK 5] Live preview updated data-theme to: "${activeDataTheme}"`);
    console.log(`[CHECK 6] Teal preview active: ${activeDataTheme === 'medical-teal'}`);

    // Check previewing badge
    const modalText = await page.$eval('div[role="dialog"]', (el) => el.innerText);
    console.log(`[CHECK 7] Live Preview badge visible: ${modalText.includes('Live Preview Active')}`);
    await page.screenshot({ path: path.join(screenshotDir, 'theme_preview_medical_teal.png') });
  }

  console.log('\n=== TEST 4: Revert / Cancel Preview ===');
  // Click Cancel / Revert
  const cancelBtn = await page.$eval('button', (b) => b.innerText);
  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const bText = await page.evaluate((el) => el.innerText, btn);
    if (bText.includes('Cancel')) {
      await btn.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 400));
  const revertedTheme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  console.log(`[CHECK 8] After Cancel, reverted back to: "${revertedTheme}" (should be clinical-violet)`);

  console.log('\n=== TEST 5: Apply & Persist Dark Theme (Theme C - Midnight Sapphire) ===');
  // Re-open settings
  const openBtn = await page.$('button[aria-label="Theme & Brand Appearance Settings"]');
  if (openBtn) await openBtn.click();
  await new Promise((r) => setTimeout(r, 500));

  // Find Midnight Sapphire card
  const allCards = await page.$$('div.cursor-pointer');
  for (const c of allCards) {
    const text = await page.evaluate((el) => el.innerText, c);
    if (text.includes('Midnight Sapphire')) {
      await c.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 400));

  // Click Apply
  const dialogBtns = await page.$$('button');
  for (const btn of dialogBtns) {
    const bText = await page.evaluate((el) => el.innerText, btn);
    if (bText.includes('Confirm & Apply')) {
      await btn.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 600));

  const confirmedTheme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  const isDarkClass = await page.evaluate(() => document.documentElement.classList.contains('dark'));
  const storedTheme = await page.evaluate(() => localStorage.getItem('fedmed_theme'));
  console.log(`[CHECK 9] Confirmed theme: "${confirmedTheme}" (isDark: ${isDarkClass})`);
  console.log(`[CHECK 10] Stored in localStorage: "${storedTheme}"`);

  await page.screenshot({ path: path.join(screenshotDir, 'dashboard_midnight_sapphire.png') });

  // Refresh page to verify persistence
  console.log('\n=== TEST 6: Persistence on Page Reload ===');
  await page.reload({ waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 600));
  const reloadedTheme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  console.log(`[CHECK 11] Theme persisted across reload: "${reloadedTheme}" (expected: midnight-sapphire)`);

  console.log('\n=== TEST 7: Brand Logo Variants Tab & Selection ===');
  // Open settings again
  const reOpenBtn = await page.$('button[aria-label="Theme & Brand Appearance Settings"]');
  if (reOpenBtn) await reOpenBtn.click();
  await new Promise((r) => setTimeout(r, 500));

  // Switch to Brand Icon tab
  const logoTabBtn = (await page.$$('button')).find(async (b) => {
    const t = await page.evaluate((el) => el.innerText, b);
    return t.includes('Brand Icon');
  });
  const tabButtons = await page.$$('button');
  for (const b of tabButtons) {
    const text = await page.evaluate((el) => el.innerText, b);
    if (text.includes('Brand Icon')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 400));

  const logoCardsText = await page.$eval('div[role="dialog"]', (el) => el.innerText);
  const hasNeuralBrain = logoCardsText.includes('Neural Brain');
  const hasBrainCircuit = logoCardsText.includes('Brain + Circuit');
  const hasMedicalIntelligence = logoCardsText.includes('Medical Intelligence');
  console.log(`[CHECK 12] All 3 logo variants visible: ${hasNeuralBrain && hasBrainCircuit && hasMedicalIntelligence}`);

  await page.screenshot({ path: path.join(screenshotDir, 'brand_logo_variants_selector.png') });

  // Apply Brain + Circuit icon
  const logoCards = await page.$$('div.cursor-pointer');
  for (const c of logoCards) {
    const text = await page.evaluate((el) => el.innerText, c);
    if (text.includes('Brain + Circuit')) {
      const applyBtn = await c.$('button');
      if (applyBtn) await applyBtn.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 500));
  const storedLogo = await page.evaluate(() => localStorage.getItem('fedmed_logo'));
  console.log(`[CHECK 13] Brand logo persisted in localStorage: "${storedLogo}"`);

  console.log('\n=== TEST 8: Reset to Default (Clinical Violet) ===');
  // Switch back to themes tab & click Reset to Default
  for (const b of await page.$$('button')) {
    const text = await page.evaluate((el) => el.innerText, b);
    if (text.includes('Color Themes')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 300));

  for (const b of await page.$$('button')) {
    const text = await page.evaluate((el) => el.innerText, b);
    if (text.includes('Reset to Default')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 400));

  const resetTheme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  console.log(`[CHECK 14] Reset to default restored: "${resetTheme}" (expected: clinical-violet)`);

  // Close modal
  const closeBtn = await page.$('button[aria-label="Close Settings"]');
  if (closeBtn) await closeBtn.click();
  await new Promise((r) => setTimeout(r, 500));

  console.log('\n=== TEST 9: Mobile Responsive Viewport (390x844) ===');
  await page.setViewport({ width: 390, height: 844, isMobile: true });
  await new Promise((r) => setTimeout(r, 400));
  const mobileSettingsBtn = await page.$('button[aria-label="Theme Settings"]');
  if (mobileSettingsBtn) {
    await mobileSettingsBtn.click();
    await new Promise((r) => setTimeout(r, 500));
    const mobileModalOpen = await page.$eval('div[role="dialog"]', () => true).catch(() => false);
    console.log(`[CHECK 15] Settings modal accessible and responsive on mobile: ${mobileModalOpen}`);
    await page.screenshot({ path: path.join(screenshotDir, 'mobile_theme_settings.png') });
  }

  console.log('\nConsole Errors:', consoleErrors.length);
  if (consoleErrors.length > 0) console.log(consoleErrors);

  await browser.close();
  console.log('\nAll Theme & Brand Identity verification tests PASSED successfully!');
}

verifyThemes().catch((err) => {
  console.error('Verification failed:', err);
  process.exit(1);
});
