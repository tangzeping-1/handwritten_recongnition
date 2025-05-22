const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let painting = false;

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener("mousedown", () => { painting = true; });
canvas.addEventListener("mouseup", () => { painting = false; ctx.beginPath(); });
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!painting) return;
    ctx.lineWidth = 15;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").innerText = "结果：-";
}

function predict() {
    const dataURL = canvas.toDataURL("image/png");

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = `结果：${data.prediction}`;
    })
    .catch(err => {
        console.error("识别失败", err);
        document.getElementById("result").innerText = "识别失败";
    });
}
