# Common
import os

# Framework
from framework.utils.io import get_select, get_text, get_integer
from framework.FxHybridTaxonomyFrameworkSetup import CFxHybridTaxonomyFrameworkSetup
from framework.FxHybridTaxonomyFramework import CFxHybridTaxonomyFramework

# Labeler
from framework.labeler.FxLabelerSetup import CFxLabelerSetup

# Embedder
from framework.embeder.FxEmbedderSetup import CFxEmbedderSetup

# Classifier
from framework.classifier.FxClassifierSetup import CFxClassifierSetup

# Searcher
from framework.searcher.FxSearcherSetup import CFxSearcherSetup






def main():
    cwd = os.getcwd()

    setup_fx = CFxHybridTaxonomyFrameworkSetup()
    setup_fx.workspace = os.path.join(cwd,"workspaces","sbert")
    setup_fx.db_source_path = "/mnt/d/Workspaces/HybridTaxonomyFramework/data/amazon/extracts/Amazon Products.db"
    setup_fx.db_working_path = os.path.join(setup_fx.workspace,"Amazon Products.db")
    setup_fx.embed_data_names = ["title","features", "description"]
    setup_fx.embed_text_formats = ["[TITLE: %s]", "[FEATURES: %s]","[DESCRIPTION: %s]"]
    fx = CFxHybridTaxonomyFramework()
    fx.setup_fx(setup_fx)

    setup_labeler = CFxLabelerSetup()
    setup_labeler.workspace = setup_fx.workspace
    setup_labeler.db_working_path = setup_fx.db_working_path
    fx.setup_labeler(setup_labeler)

    setup_embedder = CFxEmbedderSetup()
    setup_embedder.framework = "sentence-transformers"
    setup_embedder.pretrained = "all-MiniLM-L6-v2"
    # setup_embedder.pretrained = "BAAI/bge-m3"
    # setup_embedder.batch_size = 4
    setup_embedder.device = "cpu"
    setup_embedder.db_working_path = setup_fx.db_working_path
    fx.setup_embedder(setup_embedder)

    setup_classifier = CFxClassifierSetup()
    fx.setup_classifier(setup_classifier)

    setup_searcher = CFxSearcherSetup()
    fx.setup_searcher(setup_searcher)


    options = [
        "Initialize workspace and data,init",
        "Labeling,label",
        "Embedding,embed",
        "Train,train",
        "Test,test",
        "Recommend,recommend"
    ]

    while True:
        option = get_select("Select option to run:", options)
        print(f"Select: {option}")
        if option is None:
            break

        if option == "init":
            fx.initialize(force=True)
            pass

        if option == "label":
            fx.labeling()
            pass

        elif option == "embed":
            fx.embedding()
            pass

        elif option == "train":
            fx.train()
            pass

        elif option == "test":
            pass

        elif option == "recommend":
            pass
    pass

if __name__ == "__main__":
    main()
    pass