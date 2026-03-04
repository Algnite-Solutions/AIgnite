"""
Paper Annotation Website for Reranking Comparison.

A simple Flask app for blind annotation of papers that were added/removed
by personalized reranking. Users annotate without knowing the source.

Usage:
    cd scratchpad
    python annotate_app.py

    Then open http://localhost:5000
"""

import json
import random
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

import fitz  # PyMuPDF

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
COMPARISON_FILE = BASE_DIR / "comparison_Qi_Zhu.json"
PDF_CACHE_DIR = BASE_DIR / "pdfs_cache"
ANNOTATIONS_FILE = Path(__file__).parent / "annotations.json"

# Global state for annotation session
annotation_data = []
current_index = 0
annotations = []


def load_comparison_data():
    """Load comparison JSON and prepare papers for annotation."""
    global annotation_data

    with open(COMPARISON_FILE, 'r') as f:
        data = json.load(f)

    # Collect all added/removed papers per day
    for day_result in data['per_day_results']:
        day = day_result['day']
        query = day_result['query']

        papers = []

        # Add papers from "added" list
        for paper_id in day_result.get('added', []):
            papers.append({
                'day': day,
                'query': query,
                'paper_id': paper_id,
                'actual_source': 'added'  # Hidden from UI
            })

        # Add papers from "removed" list
        for paper_id in day_result.get('removed', []):
            papers.append({
                'day': day,
                'query': query,
                'paper_id': paper_id,
                'actual_source': 'removed'  # Hidden from UI
            })

        # Shuffle papers within this day to blind the source
        random.shuffle(papers)
        annotation_data.extend(papers)

    print(f"Loaded {len(annotation_data)} papers for annotation")


def get_pdf_first_page(paper_id: str) -> bytes | None:
    """Get first page of PDF as PNG bytes."""
    # Normalize paper ID (handle version suffix like v1, v2)
    normalized_id = paper_id.replace('v1', '').replace('v2', '').replace('v3', '')

    # Try different path variations
    possible_paths = [
        PDF_CACHE_DIR / f"{paper_id}.pdf",
        PDF_CACHE_DIR / f"{normalized_id}.pdf",
    ]

    pdf_path = None
    for path in possible_paths:
        if path.exists():
            pdf_path = path
            break

    if not pdf_path:
        print(f"PDF not found for {paper_id}")
        return None

    try:
        doc = fitz.open(pdf_path)
        page = doc[0]  # First page
        pix = page.get_pixmap(dpi=150)  # Reasonable resolution
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as e:
        print(f"Error rendering PDF {paper_id}: {e}")
        return None


def save_annotations():
    """Save annotations to JSON file."""
    output = {
        'user': 'Qi Zhu',
        'annotated_at': datetime.now().isoformat(),
        'total_papers': len(annotation_data),
        'annotated_count': len(annotations),
        'annotations': annotations
    }

    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(annotations)} annotations to {ANNOTATIONS_FILE}")


@app.route('/')
def index():
    """Main annotation interface."""
    global current_index

    if current_index >= len(annotation_data):
        # All papers annotated
        return render_template('complete.html',
                               total=len(annotations),
                               annotations_file=str(ANNOTATIONS_FILE))

    paper = annotation_data[current_index]
    return render_template('index.html',
                           paper=paper,
                           current=current_index + 1,
                           total=len(annotation_data))


@app.route('/pdf/<paper_id>')
def serve_pdf(paper_id):
    """Serve PDF first page as PNG image."""
    img_bytes = get_pdf_first_page(paper_id)
    if img_bytes:
        return Response(img_bytes, mimetype='image/png')
    else:
        # Return a placeholder or 404
        return Response(b'', status=404)


@app.route('/annotate', methods=['POST'])
def annotate():
    """Save annotation and advance to next paper."""
    global current_index, annotations

    data = request.json
    label = data.get('label')

    if current_index < len(annotation_data):
        paper = annotation_data[current_index]

        annotations.append({
            'day': paper['day'],
            'query': paper['query'],
            'paper_id': paper['paper_id'],
            'actual_source': paper['actual_source'],
            'label': label,
            'annotated_at': datetime.now().isoformat()
        })

        save_annotations()
        current_index += 1

    return jsonify({
        'success': True,
        'next_index': current_index,
        'total': len(annotation_data)
    })


@app.route('/skip', methods=['POST'])
def skip():
    """Skip current paper (save as 'skipped')."""
    global current_index, annotations

    if current_index < len(annotation_data):
        paper = annotation_data[current_index]

        annotations.append({
            'day': paper['day'],
            'query': paper['query'],
            'paper_id': paper['paper_id'],
            'actual_source': paper['actual_source'],
            'label': 'skipped',
            'annotated_at': datetime.now().isoformat()
        })

        save_annotations()
        current_index += 1

    return jsonify({
        'success': True,
        'next_index': current_index,
        'total': len(annotation_data)
    })


@app.route('/stats')
def stats():
    """Show annotation statistics."""
    # Count labels
    label_counts = {}
    for ann in annotations:
        label = ann['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    # Count by source
    source_counts = {'added': 0, 'removed': 0}
    for ann in annotations:
        source = ann.get('actual_source', 'unknown')
        if source in source_counts:
            source_counts[source] += 1

    return jsonify({
        'total_papers': len(annotation_data),
        'annotated': len(annotations),
        'remaining': len(annotation_data) - len(annotations),
        'label_counts': label_counts,
        'source_counts': source_counts
    })


@app.route('/reset')
def reset():
    """Reset annotation session (clear all annotations)."""
    global current_index, annotations
    current_index = 0
    annotations = []

    # Clear file
    if ANNOTATIONS_FILE.exists():
        ANNOTATIONS_FILE.unlink()

    return jsonify({'success': True, 'message': 'Session reset'})


if __name__ == '__main__':
    # Set random seed for reproducibility (remove for true randomness)
    random.seed(42)

    # Load data
    load_comparison_data()

    print(f"\n{'='*50}")
    print("Paper Annotation Website")
    print(f"{'='*50}")
    print(f"Papers to annotate: {len(annotation_data)}")
    print(f"PDF cache directory: {PDF_CACHE_DIR}")
    print(f"Annotations will be saved to: {ANNOTATIONS_FILE}")
    print(f"\nOpen http://localhost:5000 to start annotating")
    print(f"{'='*50}\n")

    app.run(debug=True, port=5000)
