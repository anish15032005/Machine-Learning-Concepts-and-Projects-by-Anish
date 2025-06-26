class XYPlotter {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.width = this.canvas.width;
    this.height = this.canvas.height;

    // Default axis limits
    this.xMin = 0;
    this.xMax = this.width;
    this.yMin = 0;
    this.yMax = this.height;
  }

  transformXY() {
    this.xMin = 0;
    this.xMax = 400;
    this.yMin = 0;
    this.yMax = 400;
  }

  toCanvasX(x) {
    return ((x - this.xMin) / (this.xMax - this.xMin)) * this.width;
  }

  toCanvasY(y) {
    return this.height - ((y - this.yMin) / (this.yMax - this.yMin)) * this.height;
  }

  plotPoint(x, y, color = "blue", radius = 2.5) {
    const cx = this.toCanvasX(x);
    const cy = this.toCanvasY(y);
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    this.ctx.fillStyle = color;
    this.ctx.fill();
  }

  plotLine(x1, y1, x2, y2, color = "black", width = 1.5) {
    const cx1 = this.toCanvasX(x1);
    const cy1 = this.toCanvasY(y1);
    const cx2 = this.toCanvasX(x2);
    const cy2 = this.toCanvasY(y2);

    this.ctx.beginPath();
    this.ctx.moveTo(cx1, cy1);
    this.ctx.lineTo(cx2, cy2);
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = width;
    this.ctx.stroke();
  }

  plotPoints(n, xArr, yArr, color = "blue") {
    for (let i = 0; i < n; i++) {
      this.plotPoint(xArr[i], yArr[i], color);
    }
  }

  drawAxes() {
    const ctx = this.ctx;
    ctx.strokeStyle = "#999";
    ctx.beginPath();
    ctx.moveTo(0, this.height);
    ctx.lineTo(this.width, this.height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, this.height);
    ctx.stroke();
  }
}
