import os
from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = 'your_secret_key'  # Replace with a secure key

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path, filter_type):
    print(f"Processing image: {image_path} with filter: {filter_type}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    # Convert to RGB for display consistency
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        # Apply filters based on practicals
        if filter_type == 'negative':
            result = 255 - image
        elif filter_type == 'mean':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pad_size = 2
            padded = np.pad(gray, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
            result = np.zeros_like(gray, dtype=np.float32)
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    region = padded[i:i+5, j:j+5]
                    result[i, j] = np.mean(region)
            result = result.astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'gaussian':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pad_size = 2
            ax = np.linspace(-pad_size, pad_size, 5)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * 1**2))
            kernel /= np.sum(kernel)
            padded = np.pad(gray, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
            result = np.zeros_like(gray, dtype=np.float32)
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    region = padded[i:i+5, j:j+5]
                    result[i, j] = np.sum(region * kernel)
            result = result.astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'median':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pad_size = 2
            padded = np.pad(gray, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
            result = np.zeros_like(gray, dtype=np.uint8)
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    region = padded[i:i+5, j:j+5]
                    result[i, j] = np.median(region)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'laplacian':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = cv2.Laplacian(gray, cv2.CV_64F)
            result = np.clip(np.abs(result), 0, 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'highpass':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            result = cv2.filter2D(gray, -1, kernel)
            result = np.clip(result, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'binary':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'otsu':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'sobel':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            result = cv2.bitwise_or(np.clip(np.abs(sobel_x), 0, 255).astype(np.uint8),
                                   np.clip(np.abs(sobel_y), 0, 255).astype(np.uint8))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'prewitt':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prewitt_x = cv2.filter2D(gray, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
            prewitt_y = cv2.filter2D(gray, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
            result = cv2.bitwise_or(np.clip(np.abs(prewitt_x), 0, 255).astype(np.uint8),
                                   np.clip(np.abs(prewitt_y), 0, 255).astype(np.uint8))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif filter_type == 'canny':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = cv2.Canny(gray, 100, 200)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            result = image_rgb  # Default to original if filter not recognized

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filter_type + '.jpg')
        print(f"Saving processed image to: {output_path}")
        if not cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR)):
            print(f"Failed to save image to {output_path}")
            return None
        return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            # Check if this is a filter request with an existing file
            filename = request.form.get('filename')
            if filename and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                filter_type = request.form.get('filter')
                if filter_type:
                    output_path = process_image(os.path.join(app.config['UPLOAD_FOLDER'], filename), filter_type)
                    if output_path:
                        return render_template('index.html', image_path=output_path, filename=filename)
                    return render_template('index.html', message='Error processing image', filename=filename)
                return render_template('index.html', message='No filter selected', filename=filename)
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"Uploaded file saved to: {file_path}")
            return render_template('index.html', image_path=file_path, filename=filename)
    
    return render_template('index.html')

@app.route('/download/<path:image_path>')
def download_file(image_path):
    return send_file(image_path, as_attachment=True)

if __name__ == '__main__':
    import os
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    port = int(os.getenv("PORT", 10000))  # Use Render's port or default to 10000
    app.run(host='0.0.0.0', port=port, debug=True)