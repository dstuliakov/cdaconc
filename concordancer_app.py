import streamlit as st
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import math
from io import BytesIO
import itertools
from functools import lru_cache

# Download necessary NLTK data
@st.cache_resource
def load_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

load_nltk_data()
lemmatizer = WordNetLemmatizer()

# --- Helper Functions ---
def parse_corpus_file(file_content, filename):
    """Extracts headline and body, calculates lengths, and preserves paragraphs."""
    prefix = filename.split('_')[0] if '_' in filename else 'UNK'
    
    headline_match = re.search(r"<Headline>(.*?)</Headline>", file_content, re.IGNORECASE | re.DOTALL)
    body_match = re.search(r"<Body>(.*?)</Body>", file_content, re.IGNORECASE | re.DOTALL)
    
    headline = headline_match.group(1).strip() if headline_match else ""
    body = body_match.group(1).strip() if body_match else ""
    full_text = headline + " " + body
    
    words = word_tokenize(full_text)
    sentences = sent_tokenize(full_text)
    paragraphs = [headline] + [p.strip() for p in body.split('\n') if p.strip()]
    
    return {
        "filename": filename,
        "prefix": prefix,
        "text": full_text,
        "headline": headline,
        "paragraphs": paragraphs,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs)
    }

def log_likelihood(c1, c2, n1, n2):
    """Calculates Log-Likelihood (G2)."""
    if c1 == 0 and c2 == 0: return 0.0
    e1 = n1 * (c1 + c2) / (n1 + n2)
    e2 = n2 * (c1 + c2) / (n1 + n2)
    ll = 0
    if c1 > 0: ll += c1 * math.log(c1 / e1)
    if c2 > 0: ll += c2 * math.log(c2 / e2)
    return 2 * ll

def get_significance(ll):
    """Returns significance stars based on Log-Likelihood."""
    if ll >= 10.83: return "***"
    elif ll >= 6.63: return "**"
    elif ll >= 3.84: return "*"
    else: return "n.s."

@lru_cache(maxsize=2048)
def is_match(corpus_token, target_token):
    """Checks if a corpus token matches the target token, supporting * wildcards and [bracketed] exact matches."""
    bracket_match = re.search(r'^(.*?)\[(.*?)\]$', target_token)
    
    if bracket_match:
        main_part = bracket_match.group(1).strip()
        extra_parts = [x.strip() for x in bracket_match.group(2).split(',')]
        
        # 1. Check main wildcard/word component
        if main_part:
            if '*' in main_part:
                pattern = "^" + re.escape(main_part).replace(r'\*', '.*') + "$"
                if re.match(pattern, corpus_token):
                    return True
            else:
                if corpus_token == main_part:
                    return True
        
        # 2. Check bracketed exact matches
        if corpus_token in extra_parts:
            return True
            
        return False
    else:
        # Standard check without brackets
        if '*' in target_token:
            pattern = "^" + re.escape(target_token).replace(r'\*', '.*') + "$"
            return bool(re.match(pattern, corpus_token))
        return corpus_token == target_token

def get_concordance(headline, paragraphs, target, lemmatize=False):
    """Extracts the exact sentence and full paragraph containing the target, supporting wildcards and brackets."""
    # Split phrase by spaces, BUT ignore spaces inside brackets [ ]
    target_words = [w for w in re.split(r'\s+(?![^\[]*\])', target.lower()) if w]
    target_len = len(target_words)
    lines = []
    
    for para in paragraphs:
        para_words = word_tokenize(para)
        search_para = [lemmatizer.lemmatize(w.lower()) if lemmatize else w.lower() for w in para_words]
        
        # Fast check: Is the target sequence in this paragraph?
        match_found = False
        for j in range(len(search_para) - target_len + 1):
            if all(is_match(search_para[j+k], target_words[k]) for k in range(target_len)):
                match_found = True
                break
                
        if not match_found: continue
            
        sentences = sent_tokenize(para)
        for sent in sentences:
            sent_words = word_tokenize(sent)
            search_sent = [lemmatizer.lemmatize(w.lower()) if lemmatize else w.lower() for w in sent_words]
            
            for j in range(len(search_sent) - target_len + 1):
                if all(is_match(search_sent[j+k], target_words[k]) for k in range(target_len)):
                    # Uppercase the matched string and wrap in markers (~~) for Excel rich text parser
                    match_str = " ".join(sent_words[j:j+target_len]).upper()
                    marked_sent = " ".join(sent_words[:j]) + f" ~~{match_str}~~ " + " ".join(sent_words[j+target_len:])
                    
                    # Construct marked paragraph
                    marked_para_list = []
                    i = 0
                    while i < len(para_words):
                        if i <= len(search_para) - target_len and all(is_match(search_para[i+k], target_words[k]) for k in range(target_len)):
                            match_str_p = " ".join(para_words[i:i+target_len]).upper()
                            marked_para_list.append(f"~~{match_str_p}~~")
                            i += target_len
                        else:
                            marked_para_list.append(para_words[i])
                            i += 1
                    marked_para = " ".join(marked_para_list)

                    lines.append({
                        "Headline": headline,
                        "Sentence with Item": marked_sent.strip(),
                        "Paragraph with Item": marked_para.strip()
                    })
    return lines

# --- Main App ---
st.title("Corpus Concordancer & Statistical Tool")

# Step 1: Upload Corpus
uploaded_files = st.file_uploader("1. Upload your corpus text files (.txt)", accept_multiple_files=True, type=['txt'])

if uploaded_files:
    # Step 2 & 3: Search Type & Input
    search_type = st.radio("2. Would you like a concordance of individual items or groups of items?", 
                           ("Individual Items", "Groups of Items"))
    
    search_dict = {}
    
    st.info("""💡 **Pro Tips:**
    - Use an asterisk (`*`) as a wildcard (e.g., `claim*` finds *claim*, *claims*, *claimed*).
    - Add square brackets for exact irregular variations (e.g., `eat*[ate, eaten]` will find *eat*, *eats*, *eating*, *ate*, and *eaten*).""")
    
    if search_type == "Individual Items":
        items_input = st.text_input("3. Enter items separated by a comma (e.g., policy, law, eat*[ate, eaten])")
        if items_input:
            # Split by comma but IGNORE commas inside brackets
            items_list = [i.strip() for i in re.split(r',\s*(?![^\[]*\])', items_input) if i.strip()]
            for item in items_list: search_dict[item] = [item]
                
    else:
        st.write("3. Define categories and items:")
        if 'num_categories' not in st.session_state:
            st.session_state.num_categories = 1
            
        for i in range(st.session_state.num_categories):
            col1, col2 = st.columns([1, 2])
            with col1:
                cat_name = st.text_input(f"Category {i+1} Name", key=f"cat_name_{i}")
            with col2:
                cat_items = st.text_input("Items (comma-separated)", key=f"cat_items_{i}")
                
            if cat_name and cat_items:
                # Split by comma but IGNORE commas inside brackets
                search_dict[cat_name.strip()] = [x.strip() for x in re.split(r',\s*(?![^\[]*\])', cat_items) if x.strip()]
                
        if st.button("➕ Add another category"):
            st.session_state.num_categories += 1
            st.rerun()

    # Step 4: Lemmatization (Updated Label)
    do_lemmatize = st.checkbox("4. Should the corpus undergo noun lemmatization (the script only lemmatizes nouns)?")
    
    st.markdown("""
    ---
    **Statistical Significance Legend (Log-Likelihood):**
    * `***` : p < 0.001 (LL ≥ 10.83)
    * `**`  : p < 0.01  (LL ≥ 6.63)
    * `*`   : p < 0.05  (LL ≥ 3.84)
    * `n.s.` : not significant
    """)

    if st.button("Generate Concordance & Statistics") and search_dict:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- File Parsing ---
        status_text.text("Parsing corpus files...")
        corpus_data = []
        total_files = len(uploaded_files)
        for i, file in enumerate(uploaded_files):
            content = file.getvalue().decode("utf-8")
            corpus_data.append(parse_corpus_file(content, file.name))
            progress_bar.progress((i + 1) / total_files * 0.3)
            
        df_corpus = pd.DataFrame(corpus_data)
        
        subcorpora = ['DT', 'GUA', 'IND', 'TIM']
        left_papers = ['GUA', 'IND']
        right_papers = ['DT', 'TIM']
        
        # --- Overall Statistics ---
        stats_dict = {
            "Metric": ["Total Wordcount", "Median Article Length (Words)", "Median Article Length (Sentences)", "Median Article Length (Paragraphs)"],
            "Entire Corpus": [
                df_corpus['word_count'].sum(), df_corpus['word_count'].median(),
                df_corpus['sentence_count'].median(), df_corpus['paragraph_count'].median()
            ]
        }
        for sub in subcorpora:
            sub_df = df_corpus[df_corpus['prefix'] == sub]
            stats_dict[sub] = [
                sub_df['word_count'].sum() if not sub_df.empty else 0,
                sub_df['word_count'].median() if not sub_df.empty else 0,
                sub_df['sentence_count'].median() if not sub_df.empty else 0,
                sub_df['paragraph_count'].median() if not sub_df.empty else 0
            ]
        df_overall_stats = pd.DataFrame(stats_dict)
        
        total_words = df_corpus['word_count'].sum()
        words_per_sub = {sub: df_corpus[df_corpus['prefix'] == sub]['word_count'].sum() for sub in subcorpora}
        words_left = df_corpus[df_corpus['prefix'].isin(left_papers)]['word_count'].sum()
        words_right = df_corpus[df_corpus['prefix'].isin(right_papers)]['word_count'].sum()

        # --- Process Frequencies & Concordance ---
        status_text.text("Calculating frequencies and extracting concordances...")
        concordance_data = []
        raw_counts = {item: {sub: 0 for sub in subcorpora} for group in search_dict.values() for item in group}
        raw_counts_grouped = {group: {sub: 0 for sub in subcorpora} for group in search_dict.keys()}
        
        for i, row in enumerate(corpus_data):
            prefix = row['prefix']
            if prefix not in subcorpora: continue
            
            for group, items in search_dict.items():
                for item in items:
                    lines = get_concordance(row['headline'], row['paragraphs'], item, lemmatize=do_lemmatize)
                    count = len(lines)
                    
                    raw_counts[item][prefix] += count
                    raw_counts_grouped[group][prefix] += count
                    
                    for line_dict in lines:
                        concordance_data.append({
                            "Lemma/Search Term": item,
                            "Category": group,
                            "Subcorpus": prefix,
                            "File": row['filename'],
                            "Headline": line_dict['Headline'],
                            "Sentence with Item": line_dict['Sentence with Item'],
                            "Paragraph with Item": line_dict['Paragraph with Item']
                        })
            progress_bar.progress(0.3 + ((i + 1) / len(corpus_data) * 0.4))

        # --- Build DataFrames ---
        df_raw_items = pd.DataFrame(raw_counts).T.fillna(0)
        df_raw_items['Total'] = df_raw_items.sum(axis=1)
        
        df_norm_items = df_raw_items.copy()
        for col in subcorpora:
            if words_per_sub.get(col, 0) > 0: df_norm_items[col] = (df_norm_items[col] / words_per_sub[col]) * 10000
        df_norm_items['Total'] = (df_norm_items['Total'] / total_words) * 10000

        df_raw_groups = pd.DataFrame(raw_counts_grouped).T.fillna(0)
        df_raw_groups['Total'] = df_raw_groups.sum(axis=1)
        
        df_norm_groups = df_raw_groups.copy()
        for col in subcorpora:
            if words_per_sub.get(col, 0) > 0: df_norm_groups[col] = (df_norm_groups[col] / words_per_sub[col]) * 10000
        df_norm_groups['Total'] = (df_norm_groups['Total'] / total_words) * 10000

        # --- Statistical Significance ---
        status_text.text("Performing statistical tests...")
        stats_comparisons = []
        pairs = list(itertools.combinations(subcorpora, 2))
        
        for item, counts in raw_counts.items():
            for p1, p2 in pairs:
                c1, c2 = counts.get(p1, 0), counts.get(p2, 0)
                n1, n2 = words_per_sub.get(p1, 0), words_per_sub.get(p2, 0)
                if n1 == 0 or n2 == 0: continue
                
                ll = log_likelihood(c1, c2, n1, n2)
                norm1, norm2 = (c1/n1)*10000, (c2/n2)*10000
                diff_pct = ((norm1 - norm2) / norm2) if norm2 > 0 else None
                
                stats_comparisons.append({
                    "Comparison": f"{p1} vs {p2}", "Item": item, "Log-Likelihood": ll,
                    "Significance": get_significance(ll), "Effect Size (% Diff)": diff_pct
                })
            
            c_left, c_right = counts.get('GUA', 0) + counts.get('IND', 0), counts.get('DT', 0) + counts.get('TIM', 0)
            if words_left > 0 and words_right > 0:
                ll_lr = log_likelihood(c_left, c_right, words_left, words_right)
                norm_l, norm_r = (c_left/words_left)*10000, (c_right/words_right)*10000
                diff_lr = ((norm_l - norm_r) / norm_r) if norm_r > 0 else None
                
                stats_comparisons.append({
                    "Comparison": "Left (GUA+IND) vs Right (DT+TIM)", "Item": item, "Log-Likelihood": ll_lr,
                    "Significance": get_significance(ll_lr), "Effect Size (% Diff)": diff_lr
                })

        df_stats_sig = pd.DataFrame(stats_comparisons)
        df_concordance = pd.DataFrame(concordance_data)
        progress_bar.progress(0.85)

        # --- Beautiful Excel Export ---
        status_text.text("Generating Excel report...")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sheets_to_write = {
                "Corpus Stats": df_overall_stats,
                "Raw Counts (Items)": df_raw_items.reset_index(names="Item"),
                "Norm Counts (Items)": df_norm_items.reset_index(names="Item"),
            }
            if search_type == "Groups of Items":
                sheets_to_write["Raw Counts (Groups)"] = df_raw_groups.reset_index(names="Group")
                sheets_to_write["Norm Counts (Groups)"] = df_norm_groups.reset_index(names="Group")
            
            sheets_to_write["Statistical Significance"] = df_stats_sig
            sheets_to_write["Concordance Lines"] = df_concordance
            
            workbook = writer.book
            header_format = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white', 'border': 1})
            wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
            float_format = workbook.add_format({'num_format': '0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            bold_fmt = workbook.add_format({'bold': True})
            
            sig_3 = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'bold': True})
            sig_2 = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700', 'bold': True})
            sig_1 = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'bold': True})
            sig_ns = workbook.add_format({'font_color': '#808080'})

            for sheet_name, df in sheets_to_write.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                
                # Paint Headers
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Sheet-Specific Formatting
                if sheet_name == "Concordance Lines":
                    # Filter ONLY for Concordance Lines
                    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
                    
                    worksheet.set_column(0, 3, 15)
                    worksheet.set_column(4, 4, 30, wrap_format) # Headline
                    worksheet.set_column(5, 5, 50, wrap_format) # Sentence
                    worksheet.set_column(6, 6, 80, wrap_format) # Paragraph
                    
                    # Apply Rich Text (Bolding) to matches marked with '~~'
                    for row_num in range(len(df)):
                        for col_num in range(len(df.columns)):
                            cell_value = df.iloc[row_num, col_num]
                            if isinstance(cell_value, str) and '~~' in cell_value:
                                parts = cell_value.split('~~')
                                rich_text = []
                                for idx, part in enumerate(parts):
                                    if idx % 2 == 1:
                                        rich_text.extend([bold_fmt, part])
                                    else:
                                        if part: rich_text.append(part)
                                
                                if len(rich_text) > 1:
                                    worksheet.write_rich_string(row_num + 1, col_num, *rich_text, wrap_format)
                                else:
                                    worksheet.write(row_num + 1, col_num, cell_value.replace('~~', ''), wrap_format)
                            else:
                                worksheet.write(row_num + 1, col_num, cell_value, wrap_format if col_num in [4, 5, 6] else None)

                elif sheet_name == "Statistical Significance":
                    worksheet.set_column(0, 1, 30)
                    worksheet.set_column(2, 2, 15, float_format)
                    worksheet.set_column(3, 3, 15)
                    worksheet.set_column(4, 4, 20, percent_format)
                    
                    worksheet.conditional_format(1, 3, len(df), 3, {'type': 'cell', 'criteria': '==', 'value': '"***"', 'format': sig_3})
                    worksheet.conditional_format(1, 3, len(df), 3, {'type': 'cell', 'criteria': '==', 'value': '"**"', 'format': sig_2})
                    worksheet.conditional_format(1, 3, len(df), 3, {'type': 'cell', 'criteria': '==', 'value': '"*"', 'format': sig_1})
                    worksheet.conditional_format(1, 3, len(df), 3, {'type': 'cell', 'criteria': '==', 'value': '"n.s."', 'format': sig_ns})
                    worksheet.conditional_format(1, 4, len(df), 4, {'type': 'data_bar', 'bar_color': '#63C384'})
                else:
                    worksheet.set_column(0, len(df.columns) - 1, 20)

        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")
        st.success("Analysis complete!")
        
        st.download_button(
            label="📥 Download Enhanced Excel Report",
            data=output.getvalue(),
            file_name="concordance_results_enhanced.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )