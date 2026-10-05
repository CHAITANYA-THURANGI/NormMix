# 🌐 Complete Step-by-Step Guide: Publishing NormMix Website & Chrome Extension Online

This guide is designed for **first-time developers**. No prior Git, GitHub, or extension publishing experience is needed. Follow these exact steps to take your project from your local computer to the live web.

---

## 📋 Quick Cost & Platform Summary

| Component | Platform | Cost | Purpose |
| :--- | :--- | :--- | :--- |
| **Website Studio (Frontend)** | **GitHub Pages** | **100% Free** | Hosts the interactive Web Studio UI (`web/index.html`). |
| **Neural API & Model (Backend)** | **Hugging Face Spaces** | **100% Free** | Runs your Python FastAPI backend + PyTorch GPU/CPU inference with a permanent HTTPS endpoint. |
| **Extension Web Download** | **GitHub Releases & Web** | **100% Free** | Direct 1-click download of `normmix-chrome-extension-v0.5.0.zip` for any Chrome user worldwide. |
| **Official Chrome Web Store** | **Google Dev Console** | **$5 USD one-time fee** | Google charges a one-time $5 verification fee to list extensions in the public Chrome Web Store search. |

> [!NOTE]
> **Why do we separate Frontend and Backend?**
> GitHub Pages is a static web hosting service (it only serves HTML, CSS, JavaScript, and images). It cannot run Python or PyTorch directly. Therefore, we host the frontend for **free** on **GitHub Pages**, and host the Python FastAPI backend for **free** on **Hugging Face Spaces** (or Render).

---

## 🚀 PART 1: Publish the Website Online via GitHub Pages (100% Free)

### Step 1.1: Verify Your Local Git Setup
Your local repository has already been initialized and committed on your computer with your username `CHAITANYA-THURANGI` and email `chaitanyathurangi001@gmail.com`.

Open **PowerShell** and verify:
```powershell
cd C:\projects\NormMix
git status
```
You should see: `nothing to commit, working tree clean`.

---

### Step 1.2: Create a New GitHub Repository

1. Open your browser and log in to [GitHub](https://github.com/).
2. In the top-right corner, click the **`+`** icon $\to$ select **New repository**.
3. Configure the new repository:
   - **Repository name**: `NormMix` (or `telugu-english-normalizer`)
   - **Description**: `SOTA Telugu-English Code-Mixed Text Normalization & Translation Studio`
   - **Visibility**: Select **Public** (required for free GitHub Pages).
   - **Initialize this repository with**: **Leave ALL checkboxes UNCHECKED** (do NOT check "Add a README", do NOT add .gitignore, do NOT choose a license — our project already has all of these).
4. Click the green button: **Create repository**.

---

### Step 1.3: Push Your Code to GitHub

After creating the repository, GitHub will show you a page with commands under *"…or push an existing repository from the command line"*.

Copy and run these exact commands in your PowerShell terminal:

```powershell
cd C:\projects\NormMix
git remote add origin https://github.com/CHAITANYA-THURANGI/NormMix.git
git branch -M main
git push -u origin main
```

*(If prompted to authenticate with GitHub, a browser window will open asking you to click **Authorize Git Credential Manager** — sign in with your GitHub account).*

Once completed, refresh your GitHub repository page in the browser. You will see all your files!

---

### Step 1.4: Turn On GitHub Pages

1. In your GitHub repository, click on the **Settings** tab (the gear icon on the top right).
2. In the left-hand navigation sidebar, click **Pages** (under the "Code and automation" section).
3. Under **Build and deployment**:
   - **Source**: Select **Deploy from a branch**.
   - **Branch**: Click the dropdown, choose **`main`**, and leave the folder as **`/ (root)`**.
   - Click **Save**.
4. Wait about 60 to 90 seconds. Refresh the page.
5. At the top of the Pages settings, you will see a green box:
   > **"Your site is live at `https://CHAITANYA-THURANGI.github.io/NormMix/`"**

Click the link! Your NormMix Web Studio is now accessible to anyone in the world! 🎉

---

## ⚡ PART 2: Deploy Free Cloud Backend API (Hugging Face Spaces)

To allow users on the web to make live neural predictions without running Python on their own computer, deploy the FastAPI backend to **Hugging Face Spaces** for free.

### Step 2.1: Create a Free Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co/join) and create a free account.
2. Verify your email.

---

### Step 2.2: Create a New Space
1. Click on your profile picture in the top-right corner $\to$ select **New Space**.
2. Fill in the details:
   - **Space name**: `normmix-api`
   - **License**: `mit`
   - **Space SDK**: Select **Docker** (Blank)
   - **Space Hardware**: Select **Free (CPU basic · 2 vCPU · 16 GB RAM)**
   - **Visibility**: **Public**
3. Click **Create Space**.

---

### Step 2.3: Upload Backend Code to Your Space

You can push your repository code directly to your Hugging Face Space using Git:

```powershell
cd C:\projects\NormMix
git remote add hf https://huggingface.co/spaces/CHAITANYA-THURANGI/normmix-api
git push hf main
```

Hugging Face will automatically build your Docker container, start the FastAPI server, and give you a permanent public HTTPS API URL:
```
https://chaitanya-thurangi-normmix-api.hf.space
```

### Step 2.4: Connect Your Website to the Cloud Backend
Users can either:
1. Open your live website $\to$ Click the **API URL** button $\to$ Paste `https://chaitanya-thurangi-normmix-api.hf.space`.
2. Or you can update line 1203 in [`web/index.html`](file:///c:/projects/NormMix/web/index.html) so it defaults to your Hugging Face space URL for all visitors!

---

## 🧩 PART 3: Publish and Distribute the Chrome Extension

You have two distribution paths:

---

### Option A: 100% Free Direct Web Distribution (Recommended First Step)
*No fees, no approval delays. Anyone can install and use it in 60 seconds.*

#### 1. The Extension ZIP is Already Built
We created [`normmix-chrome-extension-v0.5.0.zip`](file:///c:/projects/NormMix/normmix-chrome-extension-v0.5.0.zip) (9.3 KB) and copied it directly into [`web/`](file:///c:/projects/NormMix/web/) so it is automatically hosted on your GitHub Pages website!

#### 2. Create a GitHub Release
1. In your GitHub repository, click **Releases** on the right side $\to$ **Create a new release**.
2. **Choose a tag**: Type `v0.5.0` $\to$ click **Create new tag: v0.5.0 on publish**.
3. **Release title**: `NormMix Universal Telugu-English Normalizer v0.5.0`
4. **Description**:
   ```markdown
   ### NormMix AI Chrome Extension v0.5.0
   Translate and normalize Telugu-English code-mixed text anywhere on the web!

   #### How to Install:
   1. Download `normmix-chrome-extension-v0.5.0.zip` below and unzip it.
   2. Open Chrome and go to `chrome://extensions/`.
   3. Turn ON **Developer mode** (top-right toggle switch).
   4. Click **Load unpacked** and select the unzipped folder.
   ```
5. Drag and drop `C:\projects\NormMix\normmix-chrome-extension-v0.5.0.zip` into the binary attachment box.
6. Click **Publish release**.

#### 3. Users Download from Your Website
Any user visiting `https://CHAITANYA-THURANGI.github.io/NormMix/` can click the **Chrome Extension** button in the top navigation bar to download the ZIP package and view instant installation instructions!

---

### Option B: Official Google Chrome Web Store Listing ($5 USD Fee)
*If you want your extension to appear directly in Google's official Chrome Web Store search.*

#### Step 1: Create a Chrome Web Store Developer Account
1. Go to the [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole).
2. Sign in with your Google account.
3. Pay the **one-time $5 USD registration fee** using a debit or credit card.
4. Complete your developer profile (Publisher Name, Developer Email).

#### Step 2: Upload Your Extension Package
1. On the Developer Dashboard, click **+ New Item**.
2. Drag and drop the packaged file:
   `C:\projects\NormMix\normmix-chrome-extension-v0.5.0.zip`
3. The dashboard will automatically read `manifest.json` and validate your permissions and icons.

#### Step 3: Complete Store Listing Information
- **Title**: `NormMix AI - Universal Telugu-English Normalizer`
- **Short Description**: `Normalize and translate English, Telugu, and Romanized Tanglish code-mixed text with SOTA neural AI.`
- **Detailed Description**: Describe features (auto-detection, pure Telugu, standard script, English meaning, quick sample chips).
- **Category**: Select **Productivity** or **Search Tools**.
- **Language**: English.
- **Icons**: The ZIP already contains 16x16, 48x48, and 128x128 icons.
- **Screenshots**: Take 1-2 screenshots of the extension popup (size: 1280x800 or 640x400) and upload them.

#### Step 4: Privacy & Permissions Justification
- **Single Purpose**: *"Translates and normalizes Telugu, English, and Romanized Tanglish code-mixed text in browser inputs."*
- **Permissions Justifications**:
  - `storage`: To save user's last typed text and selected engine preferences.
  - `activeTab`: To read selected text on the active webpage when the user triggers the context menu.
  - `contextMenus`: To add a right-click "Normalize with NormMix" option.

#### Step 5: Submit for Review
1. Click **Submit for Review**.
2. Google reviews and approves the extension within 24 to 48 hours.
3. Once approved, you will receive a direct Chrome Web Store link:
   `https://chromewebstore.google.com/detail/normmix-ai/...`

---

## 🛠️ Handy Maintenance Commands

When you make changes in the future and want to update your live website:

```powershell
cd C:\projects\NormMix

# 1. Check changed files
git status

# 2. Stage changes
git add .

# 3. Commit with a message
git commit -m "update: improvements to UI and models"

# 4. Push to live website
git push
```
Within 60 seconds, GitHub Pages automatically deploys your updates!

---

## ✅ Summary of Next Immediate Actions

1. Go to [github.com/new](https://github.com/new) and create the repository `NormMix` (Public, empty).
2. In PowerShell, run:
   ```powershell
   cd C:\projects\NormMix
   git remote add origin https://github.com/CHAITANYA-THURANGI/NormMix.git
   git push -u origin main
   ```
3. Go to **Settings $\to$ Pages** on GitHub, set Source to branch `main`, and click **Save**.
4. Your website is live at `https://CHAITANYA-THURANGI.github.io/NormMix/`!
