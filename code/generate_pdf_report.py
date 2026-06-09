# Generates a shareable evaluation report from the CSV output.
# This was added so retrieval experiments can be reviewed without rerunning code.


import csv
import os
import sys
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Import config settings for dynamic headers
from config import CHUNK_SIZE, CHUNK_OVERLAP, USE_MMR, MMR_LAMBDA, RERANK_THRESHOLD, REPORT_DIR

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header
        self.drawString(54, 750, "Sanskrit RAG System Evaluation Report")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        # Footer
        self.line(54, 45, 558, 45)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 30, page_text)
        self.drawString(54, 30, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.restoreState()

def build_pdf(csv_path: str, pdf_path: str):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#718096"),
        spaceAfter=20
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#2D3748")
    )
    
    table_header_text = ParagraphStyle(
        'TableHeaderText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []
    
    # Title
    story.append(Paragraph("Sanskrit RAG Evaluation Report", title_style))
    story.append(Paragraph(f"Configuration: Chunk Size={CHUNK_SIZE} verses, Overlap={CHUNK_OVERLAP} verse", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Section 1: System Config Summary
    story.append(Paragraph("1. System Configuration Parameters", section_heading))
    config_data = [
        [Paragraph("Parameter", table_header_text), Paragraph("Value", table_header_text), Paragraph("Description", table_header_text)],
        [Paragraph("Chunk Size", table_text), Paragraph(str(CHUNK_SIZE), table_text), Paragraph("Number of verses per text chunk", table_text)],
        [Paragraph("Chunk Overlap", table_text), Paragraph(str(CHUNK_OVERLAP), table_text), Paragraph("Number of overlapping verses between adjacent chunks", table_text)],
        [Paragraph("Use MMR", table_text), Paragraph(str(USE_MMR), table_text), Paragraph("Whether Maximal Marginal Relevance is used for diversity", table_text)],
        [Paragraph("Rerank Threshold", table_text), Paragraph(str(RERANK_THRESHOLD), table_text), Paragraph("Score threshold applied to CrossEncoder candidates", table_text)],
    ]
    t_config = Table(config_data, colWidths=[110, 80, 314])
    t_config.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C5282")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t_config)
    story.append(Spacer(1, 15))
    
    # Read CSV results
    query_rows = []
    summary_metrics = {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        
        for r in rows:
            if not r:
                continue
            if r[0] == "Average Value":
                # This is the summary average row
                summary_metrics = {
                    "P@5": r[1],
                    "R@5": r[2],
                    "MRR": r[3],
                    "NDCG@5": r[4],
                    "MAP": r[5]
                }
            elif r[0] == "Metric Summary" or r[0] == "":
                continue
            else:
                query_rows.append(r)
                
    # Section 2: Global Averages Summary
    story.append(Paragraph("2. Global Metric Summary (k=5)", section_heading))
    summary_data = [
        [Paragraph("Mean Precision@5", table_header_text), 
         Paragraph("Mean Recall@5", table_header_text), 
         Paragraph("MRR", table_header_text), 
         Paragraph("Mean NDCG@5", table_header_text), 
         Paragraph("MAP", table_header_text)],
        [Paragraph(summary_metrics.get("P@5", "N/A"), table_text),
         Paragraph(summary_metrics.get("R@5", "N/A"), table_text),
         Paragraph(summary_metrics.get("MRR", "N/A"), table_text),
         Paragraph(summary_metrics.get("NDCG@5", "N/A"), table_text),
         Paragraph(summary_metrics.get("MAP", "N/A"), table_text)]
    ]
    t_summary = Table(summary_data, colWidths=[100, 100, 100, 100, 104])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 15))
    
    # Section 3: Detailed Query Results
    story.append(Paragraph("3. Detailed Query-Level Performance", section_heading))
    
    query_table_data = [
        [Paragraph("ID", table_header_text),
         Paragraph("Query String", table_header_text),
         Paragraph("P@5", table_header_text),
         Paragraph("R@5", table_header_text),
         Paragraph("MRR", table_header_text),
         Paragraph("NDCG@5", table_header_text),
         Paragraph("MAP", table_header_text)]
    ]
    
    for r in query_rows:
        qid = r[0]
        q_text = r[1]
        p5 = f"{float(r[2]):.4f}"
        r5 = f"{float(r[3]):.4f}"
        mrr_val = f"{float(r[4]):.4f}"
        ndcg5 = f"{float(r[5]):.4f}"
        map_val = f"{float(r[6]):.4f}"
        
        if len(q_text) > 48:
            q_text = q_text[:45] + "..."
            
        query_table_data.append([
            Paragraph(qid, table_text),
            Paragraph(q_text, table_text),
            Paragraph(p5, table_text),
            Paragraph(r5, table_text),
            Paragraph(mrr_val, table_text),
            Paragraph(ndcg5, table_text),
            Paragraph(map_val, table_text),
        ])
        
    t_queries = Table(query_table_data, colWidths=[24, 230, 50, 50, 50, 50, 50])
    t_queries.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('ALIGN', (1,0), (1,-1), 'LEFT'),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_queries)
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF report at: {pdf_path}")

if __name__ == "__main__":
    csv_file = "report/eval_results.csv"
    pdf_file = "report/eval_results.pdf"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        pdf_file = sys.argv[2]
        
    build_pdf(csv_file, pdf_file)
