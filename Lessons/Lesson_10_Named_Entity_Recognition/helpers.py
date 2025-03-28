import pandas as pd
import re
from rapidfuzz.fuzz import token_sort_ratio
import spacy
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from spacy import displacy
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import torch


# def generate_entity_colors(full_label_list, cmap_name="Set2", special_labels=None):
#     """
#     Generate a dictionary mapping entity labels to unique colors for visualization.

#     Args:
#         full_label_list (list): A list of all possible entity labels (e.g., ["B-PER", "I-PER", "O"]).
#         cmap_name (str): Name of the matplotlib colormap to use for generating colors.
#         special_labels (dict): A dictionary of special labels and their custom colors (e.g., {"IGNORE": "#cccccc"}).

#     Returns:
#         dict: A dictionary where keys are entity labels and values are their corresponding hex color codes.
#     """
#     # Get the colormap from matplotlib
#     cmap = plt.get_cmap(cmap_name)
#     colors = {}

#     if special_labels is None:
#         special_labels = {}

#     # Assign a unique color to each label, skipping the "O" label (outside entities)
#     for i, label in enumerate(sorted(full_label_list)):
#         if label == "O":
#             continue  # "O" is not an entity, so we skip it
#         rgba = cmap(i % cmap.N)  # Get RGBA color from colormap
#         rgb = tuple(int(255 * x) for x in rgba[:3])  # Convert to RGB
#         hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)  # Convert RGB to hex
#         colors[label] = hex_color

#     # Add any custom colors for special labels
#     for label, hex_color in special_labels.items():
#         colors[label] = hex_color

#     return colors


# def display_ner_html(tokens, ner_tags, label_list,
#                      entity_colors=None, cmap_name="Set2",
#                      subword_label="IGNORE", subword_color="#cccccc"):
#     """
#     Displays Named Entity Recognition (NER) results in an HTML format with color-coded entities.

#     Args:
#         tokens (list): A list of tokens (words) in the input text.
#         ner_tags (list): A list of NER tags corresponding to the tokens (e.g., [0, 1, 2, -100]).
#         label_list (list): A list mapping tag indices to label strings (e.g., ["O", "B-PER", "I-PER"]).
#         entity_colors (dict): A dictionary mapping entity labels to hex color codes. If None, colors are auto-generated.
#         cmap_name (str): Name of the matplotlib colormap to use for generating colors (default: "Set2").
#         subword_label (str): Label to use for subword tokens (default: "IGNORE").
#         subword_color (str): Hex color code for subword tokens (default: "#cccccc").

#     Returns:
#         None: Displays the NER visualization in a Jupyter Notebook or IPython environment.
#     """
#     # Create a blank spaCy language model
#     nlp = spacy.blank("en")

#     # Create a spaCy Doc object with tokens and spaces
#     spaces = [True] * (len(tokens) - 1) + [False]  # Add spaces between tokens except the last one
#     doc = Doc(nlp.vocab, words=tokens, spaces=spaces)

#     # Convert ner_tags (indices) to BIO tag strings using the label_list
#     bio_tags = [label_list[tag] if tag != -100 else subword_label for tag in ner_tags]

#     spans = []  # List to store entity spans
#     current_entity = []  # Temporary storage for the current entity's token indices
#     current_label = None  # Current entity label

#     # Iterate through BIO tags to create entity spans
#     for i, tag in enumerate(bio_tags):
#         if tag == subword_label:
#             # Subword tokens are treated as individual spans
#             spans.append(Span(doc, i, i + 1, label=subword_label))
#         elif tag.startswith("B-"):
#             # Start of a new entity
#             if current_entity:
#                 # Save the previous entity span
#                 spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))
#             current_entity = [i, i + 1]  # Start a new entity
#             current_label = tag
#         elif tag.startswith("I-") and current_label and tag[2:] == current_label[2:]:
#             # Continuation of the current entity
#             current_entity[1] = i + 1
#         else:
#             # End of the current entity
#             if current_entity:
#                 spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))
#             current_entity = []
#             current_label = None

#     # Add the last entity span if it exists
#     if current_entity:
#         spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))

#     # Filter overlapping spans to keep only the longest ones
#     doc.ents = filter_spans(spans)

#     # Generate entity colors if not provided
#     if entity_colors is None:
#         entity_colors = generate_entity_colors(
#             full_label_list=label_list,
#             cmap_name=cmap_name,
#             special_labels={subword_label: subword_color}
#         )

#     # Options for displaCy visualization
#     options = {"colors": entity_colors}

#     # Render the NER visualization as HTML
#     html = displacy.render(doc, style="ent", jupyter=False, options=options)

#     # Wrap the HTML in a styled container for better readability
#     wrapped_html = f"""
#     <div style="
#         line-height: 1.6;
#         max-width: 120ch;
#         white-space: normal;
#         word-wrap: break-word;
#         font-family: 'Segoe UI', sans-serif;
#     ">
#         {html}
#     </div>
#     """
#     # Display the HTML in a Jupyter Notebook or IPython environment
#     display(HTML(wrapped_html))

# def predict_and_display(text, model, tokenizer, label_list, entity_colors=None, cmap_name="Set2"):
#     """
#     Tokenizes input text, runs NER prediction using the model, and visualizes results using display_ner_html.
    
#     Args:
#         text (str): Input sentence (whitespace-separated tokens).
#         model: Hugging Face token classification model (e.g., DistilBERT).
#         tokenizer: Corresponding tokenizer (must support `is_split_into_words=True`).
#         label_list (List[str]): List of label names (indexed from model output).
#         entity_colors (dict): Optional dict mapping labels to hex colors.
#         cmap_name (str): Colormap to use if entity_colors is not provided.
#     """
#     words = text.split()
#     inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True).to(model.device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
#     word_ids = inputs.word_ids(batch_index=0)

#     # Align predictions to original words (mark subwords with -100)
#     aligned_labels = []
#     previous_word_idx = None
#     for token_idx, word_idx in enumerate(word_ids):
#         if word_idx is None or word_idx == previous_word_idx:
#             aligned_labels.append(-100)
#         else:
#             aligned_labels.append(predictions[token_idx])
#         previous_word_idx = word_idx

#     # Generate default consistent colors if not provided
#     if entity_colors is None:
#         from collections import OrderedDict
#         from matplotlib import pyplot as plt

#         def generate_entity_colors(label_list, cmap_name="tab10", special_labels=None):
#             cmap = plt.get_cmap(cmap_name)
#             colors = {}
#             if special_labels is None:
#                 special_labels = {}
#             for i, label in enumerate(sorted(label_list)):
#                 if label == "O":
#                     continue
#                 rgba = cmap(i % cmap.N)
#                 rgb = tuple(int(255 * x) for x in rgba[:3])
#                 hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
#                 colors[label] = hex_color
#             for label, hex_color in special_labels.items():
#                 colors[label] = hex_color
#             return colors

#         entity_colors = generate_entity_colors(label_list, cmap_name=cmap_name, special_labels={"SUBWORD": "#cccccc"})

#     # Visualize using your shared function
#     display_ner_html(tokens=words, ner_tags=aligned_labels, label_list=label_list, entity_colors=entity_colors)


def generate_entity_colors(full_label_list, cmap_name="Set2", special_labels=None):
    """
    Generates a dictionary mapping entity labels (e.g., "B-PER", "I-LOC") to unique colors for visualization.
    This helps highlight different entity types using consistent, colorblind-friendly colors.

    Args:
        full_label_list (list of str): All possible entity labels (e.g., ["O", "B-PER", "I-PER", "B-LOC", ...]).
        cmap_name (str): Name of a matplotlib colormap to use for generating the colors (default: "Set2").
        special_labels (dict): Optional dictionary of labels to manually assign specific hex colors (e.g., {"IGNORE": "#cccccc"}).

    Returns:
        dict: A dictionary mapping each label to a hex color code (e.g., {"B-PER": "#56B4E9"}).
    """
    cmap = plt.get_cmap(cmap_name)
    colors = {}

    if special_labels is None:
        special_labels = {}

    for i, label in enumerate(sorted(full_label_list)):
        if label == "O":
            continue  # Skip "O" which means "Outside" any named entity
        rgba = cmap(i % cmap.N)
        rgb = tuple(int(255 * x) for x in rgba[:3])
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
        colors[label] = hex_color

    # Add any special manually specified colors
    for label, hex_color in special_labels.items():
        colors[label] = hex_color

    return colors


def display_ner_html(tokens, ner_tags, label_list,
                     entity_colors=None, cmap_name="Set2",
                     subword_label="IGNORE", subword_color="#cccccc"):
    """
    Visualizes NER predictions using spaCy's displacy tool.

    Args:
        tokens (list of str): List of input tokens (e.g., words in a sentence).
        ner_tags (list of int): Corresponding label indices for each token. Subword tokens should have -100.
        label_list (list of str): Mapping from label indices to BIO tag strings.
        entity_colors (dict): Optional custom color mapping for entity labels.
        cmap_name (str): Name of colormap used to generate entity colors if none provided.
        subword_label (str): Label name used to mark subword tokens (default: "IGNORE").
        subword_color (str): Color used to highlight subword tokens (default: light gray).

    Returns:
        None: Displays the rendered HTML visualization inside a Jupyter notebook or IPython environment.
    """
    nlp = spacy.blank("en")
    spaces = [True] * (len(tokens) - 1) + [False]
    doc = Doc(nlp.vocab, words=tokens, spaces=spaces)

    # Convert integer tags to BIO tag strings, replacing -100 with a placeholder label
    bio_tags = [label_list[tag] if tag != -100 else subword_label for tag in ner_tags]

    spans = []
    current_entity = []
    current_label = None

    # Build spans from BIO tag sequence
    for i, tag in enumerate(bio_tags):
        if tag == subword_label:
            spans.append(Span(doc, i, i + 1, label=subword_label))  # treat subwords as individual spans
        elif tag.startswith("B-"):
            if current_entity:
                spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))
            current_entity = [i, i + 1]
            current_label = tag
        elif tag.startswith("I-") and current_label and tag[2:] == current_label[2:]:
            current_entity[1] = i + 1
        else:
            if current_entity:
                spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))
            current_entity = []
            current_label = None

    if current_entity:
        spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))

    # Remove overlapping spans and assign to the Doc
    doc.ents = filter_spans(spans)

    # Generate consistent colors for entities if not provided
    if entity_colors is None:
        entity_colors = generate_entity_colors(
            full_label_list=label_list,
            cmap_name=cmap_name,
            special_labels={subword_label: subword_color}
        )

    options = {"colors": entity_colors}

    # Render the entities using displacy and display in notebook
    html = displacy.render(doc, style="ent", jupyter=False, options=options)

    wrapped_html = f"""
    <div style="
        line-height: 1.6;
        max-width: 120ch;
        white-space: normal;
        word-wrap: break-word;
        font-family: 'Segoe UI', sans-serif;
    ">{html}</div>
    """
    display(HTML(wrapped_html))

def predict_ner_tags(text, model, tokenizer):
    """
    Tokenizes and predicts NER tags for the given text using a Hugging Face model.

    Args:
        text (str): Input sentence (e.g., "Barack Obama was born in Hawaii").
        model: A Hugging Face token classification model (e.g., DistilBERT).
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        tokens (List[str]): Original word tokens from the input text.
        predicted_tag_ids (List[int]): One predicted tag index per word (subwords/specials skipped).
    """

    # Step 1: Split the input text into whitespace-separated words
    words = text.split()

    # Step 2: Tokenize the list of words and retain word alignment
    inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True).to(model.device)

    # Step 3: Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Step 4: Convert logits to predicted label indices
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

    # Step 5: Get word IDs for each token
    word_ids = inputs.word_ids(batch_index=0)

    # Step 6: Extract one prediction per word (first subword only)
    predicted_tag_ids = []
    seen_words = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx not in seen_words:
            predicted_tag_ids.append(int(predictions[token_idx]))
            seen_words.add(word_idx)
        # skip subwords and special tokens

    # Step 7: Return the original words and corresponding predicted tags
    return words, predicted_tag_ids

def format_ner_eval_results(results):
    entity_rows = []
    
    for key, value in results.items():
        if key.startswith("eval_") and isinstance(value, dict):
            label = key.replace("eval_", "")
            row = {"Entity": label}
            row.update({k.capitalize(): round(v, 4) for k, v in value.items()})
            entity_rows.append(row)

    df = pd.DataFrame(entity_rows)
    
    # Add overall row
    overall = {
        "Entity": "Overall",
        "Precision": round(results.get("eval_overall_precision", 0), 4),
        "Recall": round(results.get("eval_overall_recall", 0), 4),
        "F1": round(results.get("eval_overall_f1", 0), 4),
        "Accuracy": round(results.get("eval_overall_accuracy", 0), 4),
    }
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    return df

def clean_token(token):
    """
    Removes leading and trailing punctuation from a token.

    This helps standardize words like "Obama," and "Obama." so they can match "Obama".

    Args:
        token (str): A word or token from the text.

    Returns:
        str: The cleaned token with punctuation stripped from both ends.
    """
    token = re.sub(r"’s|'s$", '', token)  # remove possessive endings
    return re.sub(r'^\W+|\W+$', '', token)


def match_entity_spans(predicted_json, tokens, fuzz_threshold=90):
    """
    Matches predicted entity phrases (from LLM) to spans in a tokenized sentence.

    This function is used to compare LLM-predicted entity strings (like "Barack Obama")
    to the original token list (e.g., ["Barack", "Obama", "met", "Hawaai"]).
    
    It finds all matching spans, even if:
    - The original tokens have punctuation (like "Obama,")
    - There's a small spelling difference (like "Hawaai" vs. "Hawaii")

    Args:
        predicted_json (dict): Dictionary of predicted entity types and text phrases.
                               Example: {"PER": ["Barack Obama"], "LOC": ["Hawaii"]}
        tokens (list of str): The list of tokens from the input text.
        fuzz_threshold (int): Minimum similarity score (0–100) to consider a fuzzy match.
                              Lower threshold = more forgiving.

    Returns:
        list of tuples: A list of matched entity spans in the format:
                        (label, start_index, end_index)
                        where the span is tokens[start_index:end_index].
    """
    spans = []

    # Preprocess each token to remove punctuation and lowercase it
    cleaned_tokens = [clean_token(t).lower() for t in tokens]

    # Loop through each predicted label and its list of entity strings
    for label, phrases in predicted_json.items():
        for phrase in phrases:
            # Clean and lowercase the predicted entity phrase
            phrase_tokens = [clean_token(t).lower() for t in phrase.split()]
            phrase_len = len(phrase_tokens)

            # Slide over the token list to look for fuzzy matches
            for i in range(len(tokens) - phrase_len + 1):
                # Get a window of tokens the same length as the entity phrase
                window = cleaned_tokens[i:i + phrase_len]

                # Compare the cleaned window to the predicted phrase using fuzzy match
                match_score = token_sort_ratio(' '.join(window), ' '.join(phrase_tokens))

                # If the match is strong enough, save the span
                if match_score >= fuzz_threshold:
                    spans.append((label, i, i + phrase_len))

    return spans

def spans_to_bio_tags(tokens, spans, label_list):
    """
    Converts entity spans into BIO tag indices aligned to the given tokens.

    Assumes the span labels already match the format in label_list
    (e.g., "PER", "LOC") and constructs "B-XXX"/"I-XXX" tags as needed.

    Args:
        tokens (list of str): The list of tokens in the sentence.
        spans (list of tuples): List of (label, start_idx, end_idx) tuples.
                                Each label should match a base type in label_list
                                (e.g., "PER", "LOC", "ORG", "MISC").
        label_list (list of str): List of all valid BIO tag strings.
                                  Example: ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", ...]

    Returns:
        List[int]: A list of tag indices (integers) aligned to the tokens.
    """
    # Initialize all tags as "O"
    bio_tags = ["O"] * len(tokens)

    for label, start, end in spans:
        if 0 <= start < end <= len(tokens):
            bio_tags[start] = f"B-{label}"
            for i in range(start + 1, end):
                bio_tags[i] = f"I-{label}"
        else:
            raise ValueError(f"Invalid span indices: ({start}, {end}) for label '{label}'")

    # Convert tag strings to indices using the label_list
    tag_indices = []
    for tag in bio_tags:
        try:
            tag_indices.append(label_list.index(tag))
        except ValueError:
            raise ValueError(f"Tag '{tag}' not found in label_list: {label_list}")

    return tag_indices

    return bio_tags


def json_extractor(text):
    # Extract the JSON object from the response
    try:
        text = text.strip("```json").strip("```").strip()
        json_object = json.loads(text)
    except json.JSONDecodeError:
        json_object = {"error": "Could not parse JSON"}
    return json_object
