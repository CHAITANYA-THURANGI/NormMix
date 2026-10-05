// Background service worker for NormMix AI Chrome Extension
// Creates context menu with safe removal of prior registrations to prevent duplicate ID errors.

chrome.runtime.onInstalled.addListener(() => {
  try {
    chrome.contextMenus.removeAll(() => {
      chrome.contextMenus.create({
        id: "tecm-normalize-selection",
        title: "Normalize with NormMix AI",
        contexts: ["selection"],
      }, () => {
        // Read lastError to prevent uncaught error logging
        if (chrome.runtime.lastError) {
          // Benign error suppressed
        }
      });
    });
  } catch (e) {
    // Suppress context menu registration errors
  }
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "tecm-normalize-selection" || !info.selectionText || !tab?.id) return;
  // Content scripts cannot run on internal browser pages
  if (!tab.url || tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://") || tab.url.startsWith("edge://") || tab.url.startsWith("about:")) {
    return;
  }
  try {
    await chrome.tabs.sendMessage(tab.id, { type: "TECM_NORMALIZE_SELECTION", text: info.selectionText });
  } catch (e) {
    // Suppress unhandled connection failure on inactive tabs
  }
});
