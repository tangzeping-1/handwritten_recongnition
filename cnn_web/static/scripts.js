let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let painting = false;
ctx.lineWidth = 20;
ctx.lineCap = "round";

canvas.onmousedown = () => painting = true;
canvas.onmouseup = () => painting = false;
canvas.onmousemove = (e) => {
  if (painting) {
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
  }
};

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById('result').innerText = "结果：";
}

function predict() {
  let image = canvas.toDataURL();
  fetch("/predict", {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: image })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('result').innerText = `结果：${data.prediction}`;
  });
}
