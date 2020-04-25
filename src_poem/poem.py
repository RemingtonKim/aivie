import json
import nltk
import inflect
import random
import pattern.en as en

class PoemGenerator(object):

    def __init__(self) -> None:
        """Creates a PoemGenerator instance"""
        self.poem = []
        self.last_noun = None
        self.last_verb = None
        self.last_adj = None
        self.paths = json.load(open('../data/path.json'))['path']
        self.structure = json.load(open('../data/structure.json'))['structure']
        self.inflect = inflect.engine()
        self.lemmatizer = nltk.WordNetLemmatizer()


    def clear(self) -> None:
        """Clears PoemGenerator after every each poem generation"""
        self.poem = []
        self.last_noun = None
        self.last_verb = None
        self.last_adj = None

    def strip(self, string: str) -> str:
        """
        Args:
            string (str): the string you that will be stripped and returned
        Returns:
            str: the stripped version of string passed in
        """
        return string.strip()
        
    def verb_plurality_check(self) -> bool:
        """
        Checks whether the verb that is placed into the poem should be singular or plural

        Returns:
            bool: True if the last noun uses a plural verb. False if the last noun uses a singular verb.
        """
        if self.last_noun!=None:
            if self.last_noun.lower() == 'i' or self.last_noun.lower() == 'they' or self.last_noun.lower() == 'we' or self.last_noun.lower() == 'you' or self.inflect.singular_noun(self.last_noun.lower())!=False:
                return True
            return False
        else:
            return False

    def to_be(self, tense:str) -> str:
        """
        Conjugates the verb to be depending on last_noun
        
        Args:
            tense (str): determines the tense of the verb. 'to_be_present' for present tense. 'to_be_past' for past tense.
        Returns:
            str: conjugation of the verb 'to be'
        """

        if self.last_noun == 'i':
            return 'am' if tense == 'to_be_present' else 'was'
        else:
            if self.inflect.singular_noun(self.last_noun)!=False:
                return 'are' if tense == 'to_be_present' else 'were'
            else:
                return 'is' if tense == 'to_be_present' else 'was'

    def preprocess_word(self, word: str, part_of_speech: str) -> str:
        """
        Preprocesses the word before it is placed into the poem, ensuring proper grammar.

        Args:
            word (str): the word that you want to preprocess. ex) 'hello', 'awesome', 'walked', etc.
            part_of_speech (str): the part of speech of the word passed in. ex) 'NOUN', 'VERB', 'ADJ', 'COORD_CONJ', etc.
        Returns:
            str: the preprocessed version of the word
        """

        if part_of_speech == 'NOUN':
            self.last_noun = word
            return word
        elif part_of_speech == 'PRONOUN':
            self.last_noun = word
            return word
        elif part_of_speech == 'VERB':
            if self.verb_plurality_check():
                word = self.lemmatizer.lemmatize(word, 'v')
                self.last_verb = word
                return word
            else:
                word = en.conjugate(word, tense=en.PRESENT)
                self.last_verb = word
                return word
        elif part_of_speech == 'ADJ':
            self.last_adj = word
            return word
        elif part_of_speech == 'COORD_CONJ':
            return word
        elif part_of_speech == 'ARTICLE':
            return word
        elif part_of_speech == 'ADVERB':
            return word
        elif part_of_speech == 'COLOR':
            self.last_adj = word
            return word
        elif part_of_speech == 'PREPO':
            return word
        elif part_of_speech == 'POSS_ADJ':
            return word
        elif part_of_speech == 'LINK':
            if len(word.split()) == 1:
                if word == 'to_be_present' or word == 'to_be_past':
                    return self.to_be(word)
                if self.verb_plurality_check():
                    word = self.lemmatizer.lemmatize(word, 'v')
                    self.last_verb = word
                    return word
                else:
                    word = en.conjugate(word, tense=en.PRESENT)
                    self.last_verb = word
                    return word
            else:
                return word
        elif part_of_speech == 'INT1':
            self.last_noun = word
            return word
        elif part_of_speech == 'INT2':
            return word
        elif part_of_speech == 'PUNC':
            return word        
        else:
            raise ValueError('{} is not a valid argument for part_of_speech'.format(part_of_speech))


    def generate_poem(self) -> str:
        """Generates a random poem using one of the structures provided in 'self.structures'."""
        current_structure = self.structure[str(random.randint(0, len(self.structure)-1))].split()
        #current_structure = self.structure['20'].split()
        for i in range(len(current_structure)):
                component = current_structure[i]
                if component=='|':
                    self.poem.append('\n')
                else:
                    words = open(self.paths[component], 'r').read().splitlines()
                    word = words[random.randint(0, len(words)-1)]
                    self.poem.append(self.preprocess_word(word, component))
        self.article_corrector(current_structure)
        output = '\n'.join(map(self.strip, ' '.join(self.poem).splitlines()))
        self.clear()
        return output


    def article_corrector(self, structure: list) -> None:
        """
        Goes through the poem and changed any indefinite articles that are incorrect. For example, if the phrase 'a elephant' is present, this function changes it to 'an elephant'.

        Args:
            structure (list): the structure of the poem in the form of a list. ex) ['ARTICLE', 'NOUN', 'VERB']
        """
        assert len(self.poem) == len(structure)

        for i in range(len(structure)):
            part_of_speech = structure[i]
            if part_of_speech == 'ARTICLE':
                if self.poem[i].lower() == 'a':
                    if self.inflect.singular_noun(self.poem[i+1])!=False:
                        self.poem[i] = 'the'
                    else:
                        self.poem[i] = en.article(self.poem[i+1])
            
#Generates a random poem
if __name__ == '__main__':
    poem_gen = PoemGenerator()
    print(poem_gen.generate_poem())