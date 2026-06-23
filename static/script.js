const fileInput = document.getElementById("fileInput");
const cameraInput = document.getElementById("cameraInput");
const galleryButton = document.getElementById("galleryButton");
const cameraButton = document.getElementById("cameraButton");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");
const clearHistoryButton = document.getElementById("clearHistoryButton");
const dropzone = document.getElementById("dropzone");
const previewBox = document.getElementById("previewBox");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const resultPanel = document.getElementById("resultPanel");
const emptyState = document.getElementById("emptyState");
const predictionState = document.getElementById("predictionState");
const predictionName = document.getElementById("predictionName");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const top3List = document.getElementById("top3List");
const winnerIcon = document.getElementById("winnerIcon");
const profileIcon = document.getElementById("profileIcon");
const profileTitle = document.getElementById("profileTitle");
const profileFact = document.getElementById("profileFact");
const moodChip = document.getElementById("moodChip");
const heroFruit = document.getElementById("heroFruit");
const heroMood = document.getElementById("heroMood");
const heroSubtext = document.getElementById("heroSubtext");
const brandIcon = document.getElementById("brandIcon");
const uploadSymbol = document.getElementById("uploadSymbol");
const historyList = document.getElementById("historyList");
const toast = document.getElementById("toast");
const loadingScreen = document.getElementById("loadingScreen");
const footerSummary = document.getElementById("footerSummary");
const modeToggle = document.getElementById("modeToggle");
const modeIcon = document.getElementById("modeIcon");
const celebrationLayer = document.getElementById("celebrationLayer");

const fruitThemes = {
    blend: {
        label: "Blend",
        icon: "\u{1F347}",
        mood: "Awaiting Fruit",
        subtext: "Apple, Banana, Grape, Mango, or Strawberry",
        fact: "The app will shift color, motion, glow, and accents to match the predicted fruit."
    },
    apple: {
        label: "Apple",
        icon: "\u{1F34E}",
        mood: "Crisp Apple Mode",
        subtext: "Ruby gradients, orchard greens, and a clean prediction glow.",
        fact: "Apples are rich in fiber and their skin carries many of the fruit's antioxidants."
    },
    banana: {
        label: "Banana",
        icon: "\u{1F34C}",
        mood: "Golden Banana Mode",
        subtext: "Bright yellow energy with tropical green accents.",
        fact: "Bananas are naturally packed with potassium and quick-release carbohydrates."
    },
    grape: {
        label: "Grape",
        icon: "\u{1F347}",
        mood: "Neon Grape Mode",
        subtext: "Purple tones, cool teal accents, and vineyard-style motion.",
        fact: "Grapes contain polyphenols, plant compounds associated with rich color and flavor."
    },
    mango: {
        label: "Mango",
        icon: "\u{1F96D}",
        mood: "Tropical Mango Mode",
        subtext: "Warm orange, sunlit yellow, and fresh green details.",
        fact: "Mangoes are known for vitamin C, beta-carotene, and a naturally creamy texture."
    },
    strawberry: {
        label: "Strawberry",
        icon: "\u{1F353}",
        mood: "Berry Glow Mode",
        subtext: "Red-pink highlights with leafy green sparkle.",
        fact: "Strawberries carry vitamin C and their tiny exterior seeds are true botanical fruits."
    }
};

const state = {
    selectedFile: null,
    selectedDataUrl: "",
    lastResult: null,
    history: loadHistory()
};

function normalizeFruit(label) {
    const key = String(label || "blend").toLowerCase().replace(/[^a-z]/g, "");
    return fruitThemes[key] ? key : "blend";
}

function setTheme(fruitLabel) {
    const key = normalizeFruit(fruitLabel);
    const theme = fruitThemes[key];

    document.documentElement.dataset.fruit = key;
    [brandIcon, uploadSymbol, heroFruit, profileIcon].forEach((node) => {
        node.textContent = theme.icon;
    });

    heroMood.textContent = theme.mood;
    heroSubtext.textContent = theme.subtext;
    profileTitle.textContent = key === "blend" ? "Theme awakens after prediction" : `${theme.label} profile`;
    profileFact.textContent = theme.fact;
    moodChip.textContent = theme.label;

    document.querySelectorAll(".fruit-particle").forEach((particle, index) => {
        const icons = getParticleIcons(key);
        particle.textContent = icons[index % icons.length];
    });

    document.querySelectorAll(".fruit-runner span").forEach((fruit, index) => {
        const icons = getParticleIcons(key);
        fruit.textContent = icons[index % icons.length];
    });
}

function getParticleIcons(key) {
    if (key === "blend") {
        return ["\u{1F34E}", "\u{1F34C}", "\u{1F347}", "\u{1F96D}", "\u{1F353}"];
    }
    const icon = fruitThemes[key].icon;
    return [icon, "\u{2726}", "\u{2737}", icon, "\u{25C7}"];
}

function setMode(mode) {
    document.documentElement.dataset.mode = mode;
    modeIcon.textContent = mode === "dark" ? "\u{2600}" : "\u{263E}";
    localStorage.setItem("fruit-ai-mode", mode);
}

function showToast(message, type = "success") {
    toast.textContent = message;
    toast.classList.toggle("is-error", type === "error");
    toast.classList.add("is-visible");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => {
        toast.classList.remove("is-visible");
    }, 3200);
}

function setLoading(isLoading) {
    document.body.classList.toggle("is-loading", isLoading);
    loadingScreen.setAttribute("aria-hidden", String(!isLoading));
    predictButton.disabled = isLoading || !state.selectedFile || !window.APP_CONFIG.modelReady;
}

function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function uploadFileName(file) {
    const name = file.name || "";
    if (/\.(jpe?g|png|webp|bmp)$/i.test(name)) {
        return name;
    }

    const extensionByType = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/bmp": "bmp"
    };
    return `fruit-upload.${extensionByType[file.type] || "png"}`;
}

function validateFile(file) {
    if (!file) {
        throw new Error("Please choose a fruit image first.");
    }
    if (!file.type.startsWith("image/")) {
        throw new Error("Please upload a valid image file.");
    }
    if (file.size > 10 * 1024 * 1024) {
        throw new Error("Image is too large. Please upload an image under 10 MB.");
    }
}

function readAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Could not read image preview."));
        reader.readAsDataURL(file);
    });
}

async function setSelectedFile(file) {
    try {
        validateFile(file);
        state.selectedFile = file;
        state.selectedDataUrl = await readAsDataUrl(file);

        previewImage.src = state.selectedDataUrl;
        previewBox.classList.remove("is-empty");
        fileName.textContent = file.name || "Camera capture";
        fileSize.textContent = `${formatBytes(file.size)} image ready`;
        predictButton.disabled = !window.APP_CONFIG.modelReady;
        showToast("Image ready for prediction.");
    } catch (error) {
        showToast(error.message, "error");
    }
}

async function predictCurrentImage() {
    if (!window.APP_CONFIG.modelReady) {
        showToast(window.APP_CONFIG.modelError || "Model is not available.", "error");
        return;
    }

    if (!state.selectedFile) {
        showToast("Please choose a fruit image first.", "error");
        return;
    }

    const payload = new FormData();
    payload.append("image", state.selectedFile, uploadFileName(state.selectedFile));

    try {
        setLoading(true);
        const response = await fetch("/predict", {
            method: "POST",
            body: payload
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        state.lastResult = data;
        renderPrediction(data);
        addHistory(data);
        showToast(`${data.prediction} detected with ${data.confidence.toFixed(1)}% confidence.`);

        if (data.confidence >= 90) {
            launchCelebration();
        }
    } catch (error) {
        showToast(error.message, "error");
    } finally {
        setLoading(false);
    }
}

function renderPrediction(data) {
    const key = normalizeFruit(data.prediction);
    const theme = fruitThemes[key];
    setTheme(data.prediction);

    emptyState.hidden = true;
    predictionState.hidden = false;
    resultPanel.classList.add("has-result");

    winnerIcon.textContent = theme.icon;
    predictionName.textContent = data.prediction;
    confidenceText.textContent = `${data.confidence.toFixed(1)}%`;

    requestAnimationFrame(() => {
        confidenceBar.style.width = `${Math.max(0, Math.min(data.confidence, 100))}%`;
    });

    top3List.innerHTML = "";
    data.top3.forEach((item) => {
        const row = document.createElement("div");
        row.className = "top3-row";
        row.innerHTML = `
            <span class="top3-name">${escapeHtml(item.label)}</span>
            <span class="top3-track"><i class="top3-fill"></i></span>
            <span class="top3-value">${Number(item.confidence).toFixed(1)}%</span>
        `;
        top3List.appendChild(row);
        requestAnimationFrame(() => {
            row.querySelector(".top3-fill").style.width = `${Math.max(0, Math.min(item.confidence, 100))}%`;
        });
    });

    footerSummary.textContent = `Latest: ${data.prediction} at ${data.confidence.toFixed(1)}% on ${data.device}.`;
}

function resetWorkspace() {
    state.selectedFile = null;
    state.selectedDataUrl = "";
    state.lastResult = null;
    fileInput.value = "";
    cameraInput.value = "";
    previewImage.removeAttribute("src");
    previewBox.classList.add("is-empty");
    fileName.textContent = "No image selected";
    fileSize.textContent = "Waiting for a fruit photo";
    predictButton.disabled = true;
    emptyState.hidden = false;
    predictionState.hidden = true;
    resultPanel.classList.remove("has-result");
    confidenceBar.style.width = "0%";
    footerSummary.textContent = "No confidence summary yet.";
    setTheme("blend");
}

function addHistory(data) {
    const item = {
        id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
        prediction: data.prediction,
        confidence: data.confidence,
        top3: data.top3,
        device: data.device,
        image: state.selectedDataUrl,
        createdAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    };

    state.history.unshift(item);
    state.history = state.history.slice(0, 8);
    saveHistory();
    renderHistory();
}

function renderHistory() {
    if (!state.history.length) {
        historyList.innerHTML = '<p class="history-empty">No predictions yet.</p>';
        return;
    }

    historyList.innerHTML = "";
    state.history.forEach((item) => {
        const key = normalizeFruit(item.prediction);
        const button = document.createElement("button");
        button.className = "history-item";
        button.type = "button";
        button.innerHTML = `
            <img src="${item.image}" alt="">
            <span>
                <strong>${fruitThemes[key].icon} ${escapeHtml(item.prediction)}</strong>
                <span>${escapeHtml(item.createdAt)} | ${escapeHtml(item.device || "device")}</span>
            </span>
            <span class="history-score">${Number(item.confidence).toFixed(1)}%</span>
        `;
        button.addEventListener("click", () => replayHistory(item));
        historyList.appendChild(button);
    });
}

function replayHistory(item) {
    previewImage.src = item.image;
    previewBox.classList.remove("is-empty");
    fileName.textContent = `${item.prediction} prediction`;
    fileSize.textContent = "Replayed from history";
    renderPrediction(item);
}

function loadHistory() {
    try {
        return JSON.parse(localStorage.getItem("fruit-ai-history") || "[]");
    } catch {
        return [];
    }
}

function saveHistory() {
    try {
        localStorage.setItem("fruit-ai-history", JSON.stringify(state.history));
    } catch {
        state.history = state.history.map((item) => ({ ...item, image: "" }));
        localStorage.setItem("fruit-ai-history", JSON.stringify(state.history));
    }
}

function launchCelebration() {
    celebrationLayer.innerHTML = "";
    const colors = ["var(--fruit-primary)", "var(--fruit-secondary)", "var(--fruit-tertiary)", "#ffffff"];

    for (let index = 0; index < 44; index += 1) {
        const piece = document.createElement("span");
        piece.className = "confetti-piece";
        piece.style.left = `${Math.random() * 100}%`;
        piece.style.background = colors[index % colors.length];
        piece.style.animationDelay = `${Math.random() * 0.22}s`;
        piece.style.transform = `rotate(${Math.random() * 180}deg)`;
        celebrationLayer.appendChild(piece);
    }

    window.setTimeout(() => {
        celebrationLayer.innerHTML = "";
    }, 1500);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

galleryButton.addEventListener("click", () => fileInput.click());
cameraButton.addEventListener("click", () => cameraInput.click());
predictButton.addEventListener("click", predictCurrentImage);
resetButton.addEventListener("click", resetWorkspace);

fileInput.addEventListener("change", (event) => setSelectedFile(event.target.files[0]));
cameraInput.addEventListener("change", (event) => setSelectedFile(event.target.files[0]));

dropzone.addEventListener("click", (event) => {
    if (!event.target.closest("button")) {
        fileInput.click();
    }
});

dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
    }
});

window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
    });
});

dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length > 0) {
        const file = event.dataTransfer.files[0];
        setSelectedFile(file);
    }
});

window.addEventListener("paste", (event) => {
    const item = Array.from(event.clipboardData?.items || []).find((entry) => entry.type.startsWith("image/"));
    if (!item) return;

    const pastedFile = item.getAsFile();
    const renamedFile = new File([pastedFile], "pasted-fruit.png", { type: pastedFile.type });
    setSelectedFile(renamedFile);
});

clearHistoryButton.addEventListener("click", () => {
    state.history = [];
    saveHistory();
    renderHistory();
    showToast("Prediction history cleared.");
});

modeToggle.addEventListener("click", () => {
    const nextMode = document.documentElement.dataset.mode === "dark" ? "light" : "dark";
    setMode(nextMode);
});

const savedMode = localStorage.getItem("fruit-ai-mode");
setMode(savedMode || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
setTheme("blend");
renderHistory();

if (!window.APP_CONFIG.modelReady) {
    predictButton.disabled = true;
    showToast(window.APP_CONFIG.modelError || "Model is not available.", "error");
}
