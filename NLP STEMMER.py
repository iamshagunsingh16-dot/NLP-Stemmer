
# Statistical Spell-Checker for Twitter/Social Media Domain
# Implements the Noisy Channel Model with Unigram, Bigram, and Trigram Language Models
import re
import math
import time
import string
from collections import defaultdict, Counter
from itertools import product as iterproduct

# PHASE 1: CORPUS - A training corpus that looks like Twitter
# Data Acquisition 

TWITTER_CORPUS = """
just woke up and i am so tired today i need coffee
going to the gym today feeling motivated
cant believe how fast time flies when you are having fun
just got back from a long run feeling great
the weather today is absolutely beautiful outside
i love spending time with my friends and family
working from home has its challenges but also benefits
just finished watching a great movie highly recommend
feeling happy and grateful for everything in my life
social media can be both amazing and overwhelming sometimes
just ate the best pizza i have ever had in my life
trying to learn something new every single day
the weekend is finally here time to relax and recharge
just booked tickets for a concert so excited about it
technology keeps changing so fast it is hard to keep up
love when things just come together perfectly
cant wait for the summer holidays to begin
started a new book and i cannot put it down
the coffee shop on the corner makes the best lattes
just got a new phone and i am loving it so far
feeling a bit under the weather today staying home
need to clean my apartment but i keep procrastinating
just saw the most beautiful sunset from my window
working on a big project this week wish me luck
the internet was down all morning so frustrating
just discovered a great new restaurant nearby
morning runs are the best way to start the day
been trying to eat healthier this month going well
just joined a new gym class and it was intense
the traffic today was absolutely terrible everywhere
finally finished my assignment feeling relieved
need to catch up on so many messages and emails
just had the most productive day in a long time
trying to spend less time on my phone this week
the new update broke everything so annoying
just got promoted at work feeling really proud
cooking a new recipe tonight hope it turns out well
the library is my favorite place to focus and study
spent the afternoon gardening it was so relaxing
just saw a great film at the cinema tonight
been listening to the same song on repeat all day
the meeting today went really well great news
just got back from a trip feeling refreshed
need more hours in the day to get everything done
the coffee was cold again at the office today
just signed up for an online course very excited
feeling overwhelmed but taking it one step at a time
the dog park was so fun this afternoon
just ordered takeout because cooking felt too hard
been thinking about switching careers lately
the sunset last night was absolutely breathtaking
just found twenty dollars in an old jacket pocket
need to call my parents more often miss them
the new season of my favorite show is amazing
just spilled coffee on my keyboard great start
been trying meditation and it actually helps a lot
the bus was late again this morning so annoying
just got a haircut feeling fresh and clean
working on a presentation for tomorrow wish me luck
the park was really crowded this afternoon
just baked cookies and they turned out perfect
need to buy groceries but keep forgetting to go
the alarm did not go off this morning disaster
just finished a really good book so satisfying
been trying to wake up earlier every morning
the concert last night was absolutely incredible
just adopted a cat from the shelter so happy
need to fix my sleep schedule it is all messed up
the weather changed so quickly today unexpected rain
just started journaling and it feels really good
been so busy lately barely any time for myself
the food delivery took over an hour tonight
just went for a long walk to clear my head
need to reply to those emails from last week
the new coffee blend is so much better honestly
just had brunch with old friends so wonderful
been listening to podcasts during my commute lately
the wifi keeps dropping at home so annoying
just got tickets to see my favorite band live
need to drink more water every single day
the neighbor got a new puppy so cute honestly
just finished decorating the apartment looks great
been feeling really motivated and energized lately
the pharmacy was out of stock so inconvenient
just started a new workout routine feeling it
need to schedule a dentist appointment this week
the grocery store was packed on a saturday
just made homemade pasta for the first time
been binge watching a new series this weekend
the library book is due and i forgot to return it
just got a really nice compliment today made my day
need to take a proper vacation soon i am exhausted
""".strip()

# PHASE 1.2: CORPUS PROCESSING (Data Cleaning)

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

def build_vocabulary(tokens, min_freq=1):
    freq = Counter(tokens)
    vocab = {word for word, count in freq.items() if count >= min_freq}
    vocab.add('<UNK>')
    vocab.add('<START>')
    vocab.add('<END>')
    return vocab, freq

# PHASE 2: LANGUAGE MODEL GENERATION

class UnigramModel:
    
    def __init__(self, tokens, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.total_tokens = len(tokens)
        self.counts = Counter(tokens)
    
    def probability(self, word):
        """P(word) with Laplace smoothing."""
        count = self.counts.get(word, 0)
        return (count + 1) / (self.total_tokens + self.vocab_size)
    
    def log_probability(self, word):
        return math.log(self.probability(word))


class BigramModel:
    
    def __init__(self, tokens, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.unigram_counts = Counter(tokens)
        self.bigram_counts = defaultdict(Counter)
        
        padded = ['<START>'] + tokens + ['<END>']
        for i in range(len(padded) - 1):
            self.bigram_counts[padded[i]][padded[i+1]] += 1
    
    def probability(self, word, prev_word):
        """P(word | prev_word) with Laplace smoothing."""
        bigram_count = self.bigram_counts[prev_word][word]
        prev_count = self.unigram_counts.get(prev_word, 0)
        return (bigram_count + 1) / (prev_count + self.vocab_size)
    
    def log_probability(self, word, prev_word):
        return math.log(self.probability(word, prev_word))


# PHASE 2: ERROR MODEL (Edit Distance + Confusion Matrix)

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + cost   # substitution
            )
            # Transposition
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost)
    return dp[m][n]

def error_probability(typo, word):
    if typo == word:
        return 1.0
    dist = edit_distance(typo, word)
    if dist > 3:
        return 1e-10
    return 1.0 / (10 ** dist)

def generate_candidates(word, vocab, max_edit_dist=2):
    candidates = set()
    word_lower = word.lower()
    
    # Always include original word
    if word_lower in vocab:
        candidates.add(word_lower)
    
    # Find vocab words within edit distance
    for v_word in vocab:
        if v_word.startswith('<'):
            continue
        if abs(len(v_word) - len(word_lower)) > max_edit_dist:
            continue
        if edit_distance(word_lower, v_word) <= max_edit_dist:
            candidates.add(v_word)
    
    # Fallback: return original if no candidates found
    if not candidates:
        candidates.add(word_lower)
    
    return candidates

# PHASE 2: NOISY CHANNEL SPELL CHECKER

class NoisyChannelSpellChecker:
    # Spell checker using the Noisy Channel Model.
    # word* = argmax_{w in Candidates} P(typo|word) * P(word)
    
    def __init__(self, language_model, model_type='unigram'):
        self.lm = language_model
        self.model_type = model_type
        
    def correct_word(self, word, prev_word=None, vocab=None):
        if vocab is None:
            vocab = self.lm.vocab
        
        word_lower = word.lower()
        
        # Skip correction if word is in vocabulary
        if word_lower in vocab and word_lower not in string.punctuation:
            return word_lower
        
        candidates = generate_candidates(word_lower, vocab)
        
        best_word = word_lower
        best_score = float('-inf')
        
        for candidate in candidates:
            p_error = error_probability(word_lower, candidate)
            
            if self.model_type == 'unigram':
                log_p_word = self.lm.log_probability(candidate)
            elif self.model_type == 'bigram' and prev_word:
                log_p_word = self.lm.log_probability(candidate, prev_word)
            else:
                log_p_word = math.log(1e-10)
            
            score = math.log(p_error + 1e-10) + log_p_word
            
            if score > best_score:
                best_score = score
                best_word = candidate
        
        return best_word
    
    def correct_sentence(self, sentence, vocab=None):
        if vocab is None:
            vocab = self.lm.vocab
        
        words = sentence.lower().split()
        corrected = []
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^a-z]', '', word)
            if not clean_word:
                corrected.append(word)
                continue
            
            prev_word = corrected[i-1] if i > 0 else '<START>'
        
            correction = self.correct_word(clean_word, prev_word,  vocab)
            corrected.append(correction)
        
        return ' '.join(corrected)

# PHASE 3: TEST SET (Simulated Twitter typos)

TEST_SET = [
    # (misspelled_sentence, correct_sentence, focus_word_index)
    ("i ned cofee this moring", "i need coffee this morning", [1, 2, 3]),
    ("the wether today is gret", "the weather today is great", [1, 4]),
    ("cant beleve how fsat time flyes", "cant believe how fast time flies", [1, 3, 4]),
    ("goin to the gym toady", "going to the gym today", [0, 4]),
    ("feelig so happpy and grattful", "feeling so happy and grateful", [0, 2, 4]),
    ("just finishewd wathcing a geat moive", "just finished watching a great movie", [1, 2, 4, 5]),
    ("workng from hoem is chalengng", "working from home is challenging", [0, 2, 4]),
    ("the sunst last nigt was beutiful", "the sunset last night was beautiful", [1, 3, 5]),
    ("spned the afternnon gardenig", "spent the afternoon gardening", [0, 2, 3]),
    ("ned to drnik mroe watre evry day", "need to drink more water every day", [0, 2, 3, 4, 5]),
]


def evaluate_spell_checker(checker, test_set, vocab):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    results = []
    
    for misspelled, correct, error_indices in test_set:
        mis_words = misspelled.lower().split()
        cor_words = correct.lower().split()
        corrected_words = checker.correct_sentence(misspelled, vocab).split()
        
        for i in range(min(len(mis_words), len(cor_words), len(corrected_words))):
            mis_w = re.sub(r'[^a-z]', '', mis_words[i])
            cor_w = re.sub(r'[^a-z]', '', cor_words[i])
            corr_w = re.sub(r'[^a-z]', '', corrected_words[i]) if i < len(corrected_words) else ''
            
            is_error = (mis_w != cor_w)
            was_corrected = (corr_w == cor_w)
            made_change = (corr_w != mis_w)
            
            if is_error and was_corrected:
                true_positives += 1
            elif not is_error and made_change:
                false_positives += 1
            elif is_error and not was_corrected:
                false_negatives += 1
        
        results.append({
            'input': misspelled,
            'expected': correct,
            'got': ' '.join(corrected_words)
        })
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives,
        'results': results
    }


def time_complexity_test(checkers, vocab, sample_sentences, runs=3):
    timing = {}
    for name, checker in checkers.items():
        times = []
        for _ in range(runs):
            start = time.time()
            for sent in sample_sentences:
                checker.correct_sentence(sent, vocab)
            elapsed = time.time() - start
            times.append(elapsed)
        avg_time = sum(times) / len(times)
        timing[name] = round(avg_time, 4)
    return timing

# MAIN EXECUTION

def main():
    print("=" * 60)
    print("  Statistical Spell-Checker: Twitter/Social Media Domain")
    print("  Noisy Channel Model Implementation")
    print("=" * 60)

    # Corpus Processing
    print("\n[1] Processing Twitter-style corpus...")
    tokens = clean_and_tokenize(TWITTER_CORPUS)
    vocab, freq = build_vocabulary(tokens)
    print(f"    Tokens: {len(tokens)}, Vocabulary size: {len(vocab)}")

    # Build Language Models
    print("\n[2] Building Language Models...")
    unigram = UnigramModel(tokens, vocab)
    bigram  = BigramModel(tokens, vocab)
    print("    Unigram, Bigram models built with Laplace smoothing.")

    #  Build Spell Checkers 
    checker_uni = NoisyChannelSpellChecker(unigram, model_type='unigram')
    checker_bi  = NoisyChannelSpellChecker(bigram,  model_type='bigram')
    checkers = {
        'Unigram': checker_uni,
        'Bigram':  checker_bi
    }

    # Demo Corrections
    print("\n[3] Demo Corrections:")
    demo_sentences = [
        "i ned cofee this moring",
        "workng from hoem is chalengng",
        "feelig so happpy and grattful",
    ]
    for sent in demo_sentences:
        print(f"\n  Input   : {sent}")
        print(f"  Unigram : {checker_uni.correct_sentence(sent, vocab)}")
        print(f"  Bigram  : {checker_bi.correct_sentence(sent, vocab)}")

    #  Evaluation 
    print("\n[4] Evaluation on Test Set:")
    print(f"  {'Model':<10} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 55)

    eval_results = {}
    for name, checker in checkers.items():
        metrics = evaluate_spell_checker(checker, TEST_SET, vocab)
        eval_results[name] = metrics
        print(f"  {name:<10} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f} {metrics['f1']:>10.4f} "
              f"{metrics['tp']:>5} {metrics['fp']:>5} {metrics['fn']:>5}")

    # Sample Results
    print("\n[5] Sample Sentence Corrections (Bigram Model):")
    for r in eval_results['Bigram']['results'][:4]:
        print(f"  Input   : {r['input']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Got     : {r['got']}")
        print()

    #Time Complexity
    print("[6] Time Complexity (avg seconds for full test set, 3 runs):")
    sample_sents = [row[0] for row in TEST_SET]
    timing = time_complexity_test(checkers, vocab, sample_sents)
    for name, t in timing.items():
        print(f"  {name:<10}: {t:.4f}s")

    # Summary
    print("\n[7] Summary:")
    best_model = max(eval_results, key=lambda k: eval_results[k]['f1'])
    print(f"  Best model by F1-score: {best_model} (F1={eval_results[best_model]['f1']:.4f})")
    print("\nDone.")


if __name__ == '__main__':
    main()
