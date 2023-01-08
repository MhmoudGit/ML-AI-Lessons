// linear regression
let xVals = []  
let yVals = []
let m, b

const learningRate = 0.5
const optmizer = tf.train.sgd(learningRate)

function setup() {
	createCanvas(400, 400)
	m = tf.variable(tf.scalar(random(1)))
	b = tf.variable(tf.scalar(random(1)))
}


function loss(pred, labels) {
	return pred.sub(labels).square().mean()
}

function predict(x) {
	const xs = tf.tensor1d(x)
	//y = mx + b
	const ys = xs.mul(m).add(b)
	return ys
}

function mousePressed() {
	const x = map(mouseX, 0, width, 0, 1)
	const y = map(mouseY, 0, height, 1, 0)
	xVals.push(x)
	yVals.push(y)
}

function draw() {
    tf.tidy(() => {if (xVals.length > 0) {
		const ys = tf.tensor1d(yVals)
		optmizer.minimize(() => loss(predict(xVals), ys))
	}})
	

	background(0)
	stroke(255)
	strokeWeight(5)
	for (let i = 0; i < xVals.length; i++) {
		const px = map(xVals[i], 0, 1, 0, width)
		const py = map(yVals[i], 0, 1, height, 0)
		point(px, py)
	}

    tf.tidy(() => {
        const xx = [0,1]
        const yy = predict(xx)
    
        let x1 = map(xx[0], 0, 1,0, width)
        let x2 = map(xx[1], 0, 1,0, width)
    
        let lineY = yy.dataSync()
        let y1 = map(lineY[0], 0, 1,height, 0)
        let y2 = map(lineY[1], 0, 1,height, 0)
    
        strokeWeight(1)
        line(x1, y1, x2, y2)

    })

}
