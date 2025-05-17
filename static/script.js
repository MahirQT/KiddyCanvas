document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const topPrediction = document.getElementById('topPrediction');
    const otherPredictions = document.getElementById('otherPredictions');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Initialize speech synthesis
    const synth = window.speechSynthesis;
    let isSpeaking = false;

    // Function to speak text
    function speakText(text) {
        // Cancel any ongoing speech
        if (isSpeaking) {
            synth.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        
        // Configure speech settings
        utterance.rate = 0.6;  // Slower rate (was 0.8)
        utterance.pitch = 1;
        utterance.volume = 1;

        // Add a small pause between words for better clarity
        utterance.text = text.split(' ').join(' , ');

        // Handle speech events
        utterance.onstart = () => {
            isSpeaking = true;
        };
        
        utterance.onend = () => {
            isSpeaking = false;
        };

        // Speak the text
        synth.speak(utterance);
    }

    // Function to format text for speech
    function formatTextForSpeech(char) {
        // If it's a number, say "number" before the digit
        if (/^\d$/.test(char)) {
            return `number ${char}`;
        }
        // If it's a letter, say "letter" before the character
        else if (/^[a-zA-Z]$/.test(char)) {
            return `letter ${char}`;
        }
        // For other characters, just say the character
        return char;
    }

    // Set canvas background to white
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Drawing settings
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'black';

    // Drawing functions
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getMousePos(canvas, e);
    }

    function stopDrawing() {
        isDrawing = false;
    }

    function draw(e) {
        if (!isDrawing) return;
        
        const [currentX, currentY] = getMousePos(canvas, e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
        
        [lastX, lastY] = [currentX, currentY];
    }

    // Get mouse position relative to canvas
    function getMousePos(canvas, e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        return [
            (e.clientX - rect.left) * scaleX,
            (e.clientY - rect.top) * scaleY
        ];
    }

    // Touch event handlers
    function handleTouchStart(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }

    function handleTouchMove(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }

    function handleTouchEnd(e) {
        e.preventDefault();
        const mouseEvent = new MouseEvent('mouseup', {});
        canvas.dispatchEvent(mouseEvent);
    }

    // Event listeners for mouse
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Event listeners for touch
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', handleTouchEnd, { passive: false });

    // Modify the prediction button click handler
    predictBtn.addEventListener('click', async () => {
        try {
            // Get the image data from canvas
            const imageData = canvas.toDataURL('image/png');
            
            // Show loading state
            topPrediction.innerHTML = `
                <span class="character">...</span>
                <span class="confidence">Processing...</span>
            `;
            otherPredictions.innerHTML = '';

            // Send to server
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();

            if (data.success) {
                // Update top prediction
                const topPred = data.predictions[0];
                topPrediction.innerHTML = `
                    <span class="character">${topPred.character}</span>
                    <span class="confidence">${topPred.confidence.toFixed(1)}%</span>
                `;

                // Speak the top prediction
                const speechText = formatTextForSpeech(topPred.character);
                speakText(speechText);

                // Update other predictions
                otherPredictions.innerHTML = data.predictions
                    .slice(1)
                    .map(pred => `
                        <div class="prediction-item">
                            <span>${pred.character}</span>
                            <span>${pred.confidence.toFixed(1)}%</span>
                        </div>
                    `)
                    .join('');
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('Error:', error);
            topPrediction.innerHTML = `
                <span class="character">!</span>
                <span class="confidence">Error: ${error.message}</span>
            `;
            otherPredictions.innerHTML = '';
            speakText('Error occurred');
        }
    });

    // Add click handlers for other predictions to speak them when clicked
    otherPredictions.addEventListener('click', (e) => {
        const predictionItem = e.target.closest('.prediction-item');
        if (predictionItem) {
            const char = predictionItem.querySelector('span:first-child').textContent;
            const speechText = formatTextForSpeech(char);
            speakText(speechText);
        }
    });

    // Modify clear button to stop any ongoing speech
    clearBtn.addEventListener('click', () => {
        if (isSpeaking) {
            synth.cancel();
        }
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        topPrediction.innerHTML = `
            <span class="character">-</span>
            <span class="confidence">Draw a character</span>
        `;
        otherPredictions.innerHTML = '';
    });
}); 