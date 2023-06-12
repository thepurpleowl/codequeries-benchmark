import importlib
import sys
sys.path.insert(0, './contexts')
from basecontexts import CLASS_FUNCTION, CLASS_OTHER, BaseContexts, ContextRetrievalError
from tree_sitter import Language, Parser
from collections import namedtuple

Span = namedtuple('Span', 'start_line start_col end_line end_col')

PY_LANGUAGE = Language("./my-languages.so", "python")
tree_sitter_parser = Parser()
tree_sitter_parser.set_language(PY_LANGUAGE)

NON_DISTRIBUTABLE_QUERIES = {
    'Unused import': 'UnusedImport',
    '`__eq__` not overridden when adding attributes': 'DefineEqualsWhenAddingAttributes',
    'Use of the return value of a procedure': 'UseImplicitNoneReturnValue',
    'Wrong number of arguments in a call': 'WrongNumberArgumentsInCall',
    'Comparison using is when operands support `__eq__`': 'IncorrectComparisonUsingIs',
    'Non-callable called': 'NonCallableCalled',
    '`__init__` method calls overridden method': 'InitCallsSubclassMethod',
    'Signature mismatch in overriding method': 'SignatureOverriddenMethod',
    'Conflicting attributes in base classes': 'ConflictingAttributesInBaseClasses',
    'Inconsistent equality and hashing': 'EqualsOrHash',
    'Flask app is run in debug mode': 'FlaskDebug',
    'Wrong number of arguments in a class instantiation': 'WrongNumberArgumentsInClassInstantiation',
    'Incomplete ordering': 'IncompleteOrdering',
    'Missing call to `__init__` during object initialization': 'MissingCallToInit',
    '`__iter__` method returns a non-iterator': 'IterReturnsNonIterator'
}

QUERIES_USING_AUX_RESULT = {
    'Unused import', 'Use of the return value of a procedure', 'Wrong number of arguments in a call'
}

QUERIES_IGNORE_SF_ADD = ["`__eq__` not overridden when adding attributes",
                         "Special method has incorrect signature",
                         "Non-standard exception raised in special method",
                         "Should use a 'with' statement",
                         "Modification of parameter with default"
                         ]

QUERIES_ADD_SF_AS_CONTEXT = ["Variable defined multiple times",
                             "Module is imported more than once"
                             ]

__columns__ = ["Name", "Description", "Severity", "Message",
               "Path", "Start_line", "Start_column",
               "End_line", "End_column"]


def is_relevant_block_present(context_blocks):
    """
    This sanity check function checks if the returned context Blocks
    have at least a relevant Block.
    Args:
        context_blocks : List of Context Blocks returned from Context
                         Retrieval mechanism
    Returns:
        Boolean representing presence of a relevant Block in the list
    """
    for block in context_blocks:
        if(block.relevant):
            return True
    return False


def are_same_block(context_block, supporting_fact_block):
    return (context_block.start_line == supporting_fact_block.start_line
            and context_block.end_line == supporting_fact_block.end_line
            and context_block.other_lines == supporting_fact_block.other_lines)


def get_class_linkage(local_class, sf_block_class, local_class_mro, sf_block_class_mro):
    """
    This function retunrs class linkage to make the context continuous.
    Args:
        local_class: local class name
        sf_block_class: supporting fact class name
        local_class_mro: MRO of local_class
        sf_block_class_mro: MRO of sf_block_class
    Returns:
        Class heirarchy or inheritance order between local_class and sf_block_class
    """
    assert local_class == local_class_mro[0]
    assert sf_block_class == sf_block_class_mro[0]

    linkage_classes = None
    # if sf comes from super class of local block
    if(sf_block_class in local_class_mro
            and local_class not in sf_block_class_mro):
        sf_index = local_class_mro.index(sf_block_class)
        linkage_classes = local_class_mro[0:sf_index + 1]
    # elif sf comes from subclass of local block
    elif(sf_block_class not in local_class_mro
            and local_class in sf_block_class_mro):
        local_index = sf_block_class_mro.index(local_class)
        linkage_classes = sf_block_class_mro[0:local_index + 1]

    return linkage_classes


def add_sf_block_with_class_heirarchy_check(program_content, supporting_fact_block, adjusted_result_span):
    """
    This function checks for class linkage for a supporting block which
    is not yet added to context throught context extraction process.
    Args:
        program_content: source code
        supporting_fact_block: considered supporting fact block
        adjusted_result_span: CodeQL result span
    Returns:
        Tuple with first value being whether the second element blocks
        should be added to context.
    """
    check_add_block_types = [CLASS_FUNCTION, CLASS_OTHER]
    # if supporting fact block is of type MODULE_OTHER/MODULE_FUNCTION
    if(supporting_fact_block.block_type not in check_add_block_types):
        return True, [supporting_fact_block]

    required_sf_blocks = []
    local_block_start_line = adjusted_result_span.start_line
    local_block_end_line = adjusted_result_span.end_line

    context_object = BaseContexts(tree_sitter_parser)
    local_block = context_object.get_local_block(program_content, local_block_start_line, local_block_end_line)
    local_class_block = context_object.get_local_class(program_content,
                                                       local_block_start_line,
                                                       local_block_end_line)
    sf_local_class_block = context_object.get_local_class(program_content,
                                                          supporting_fact_block.start_line,
                                                          supporting_fact_block.end_line)
    all_blocks = context_object.get_all_blocks(program_content)

    local_class = local_class_block.metadata.split('.')[-1]
    sf_block_class = sf_local_class_block.metadata.split('.')[-1]
    # elif local block is some module level block
    if(local_block.block_type not in check_add_block_types):
        for block in all_blocks:
            if (block.start_line >= sf_local_class_block.start_line
                    and block.end_line <= sf_local_class_block.end_line):
                if(are_same_block(block, supporting_fact_block)):
                    required_sf_blocks.append(supporting_fact_block)  # relevancy maintained
                elif(block.block_type == CLASS_FUNCTION
                        and block.metadata.split('.')[-1] == '__init__'
                        and block.metadata.split('.')[-2] == sf_block_class):
                    block.relevant = True
                    required_sf_blocks.append(block)
                elif(block.block_type == CLASS_OTHER
                        and block.metadata.split('.')[-1] == sf_block_class):
                    block.relevant = True
                    required_sf_blocks.append(block)
        return True, required_sf_blocks
    # elif both supporting fact and local block are some class blocks
    else:
        local_class_mro = context_object.get_class_MRO(program_content, local_class)
        sf_block_class_mro = context_object.get_class_MRO(program_content, sf_block_class)
        linkage_classes = get_class_linkage(local_class, sf_block_class,
                                            local_class_mro, sf_block_class_mro)
        if(linkage_classes is None):
            return True, [supporting_fact_block]
        else:
            for block in all_blocks:
                if(are_same_block(block, supporting_fact_block)):
                    required_sf_blocks.append(supporting_fact_block)  # relevancy maintained
                elif(block.block_type == CLASS_FUNCTION
                        and block.metadata.split('.')[-1] == '__init__'
                        and block.metadata.split('.')[-2] in linkage_classes):
                    block.relevant = True
                    required_sf_blocks.append(block)
                elif(block.block_type == CLASS_OTHER
                        and block.metadata.split('.')[-1] in linkage_classes):
                    block.relevant = True
                    required_sf_blocks.append(block)
        return True, required_sf_blocks


def get_span_context(name, program_content, parser, file_path, message, result_span, aux_result_df):
    """
    This is helper function to get query specific context/Block given
    program_content, start_line, end_line and name of the query.
    Args:
        name : name of the query
        program_content: Program in string format from which we need to
        extract classes
        parser : tree_sitter_parser
        file_path : file of program_content
        message : CodeQL message
        result_span: CodeQL-treesitter adjusted namedtuple of
                     (start_line, start_col, end_line, end_col)
        aux_result_df: auxiliary query results dataframe
    Returns:
        A list consisting context Blocks with relevance label
    """
    if name in NON_DISTRIBUTABLE_QUERIES.keys():
        context_class = NON_DISTRIBUTABLE_QUERIES[name]
        # module = 'contexts.' + context_class.lower()  # module relative import
        module = context_class.lower()
    else:
        # module = 'contexts.' + 'distributable'
        module = 'distributable'

    # context_module = importlib.import_module(module, package=__name__.split('.')[0])
    context_module = importlib.import_module(module)

    try:
        if(name in QUERIES_USING_AUX_RESULT):
            context_blocks = context_module.get_query_specific_context(program_content, parser,
                                                                       file_path, message,
                                                                       result_span, aux_result_df)
        else:
            context_blocks = context_module.get_query_specific_context(program_content, parser,
                                                                       file_path, message, result_span)

        if(len(context_blocks) == 0):
            # this might happen because of wrong flagged span
            # those specific flags are ignored during labeling
            raise ContextRetrievalError({"message": "No context Block",
                                         "type": "empty_context/irrelevant_context"})
        elif(not is_relevant_block_present(context_blocks)):
            raise ContextRetrievalError({"message": "No relevant context Block",
                                         "type": "empty_context/irrelevant_context"})

        return context_blocks

    except ContextRetrievalError as error:
        error_att = error.args[0]
        err_msg = error_att['message']
        err_type = error_att['type']
        raise ContextRetrievalError({"message": err_msg,
                                     "type": err_type})
    except Exception as error:
        raise ContextRetrievalError({"message": str(error),
                                     "type": "generic"})


def get_local_block_context(program_content, result_span):
    """
    This is helper function to get query specific context/Block given
    program_content, start_line, end_line and name of the query.
    Args:
        program_content: Program in string format from which we need to
        extract classes
        result_span: CodeQL-treesitter adjusted namedtuple of
                     (start_line, start_col, end_line, end_col)
    Returns:
        Local Blocks with relevance label
    """
    context_object = BaseContexts(tree_sitter_parser)

    try:
        local_block = context_object.get_local_block(program_content,
                                                     result_span.start_line,
                                                     result_span.end_line)

        if(local_block is None):
            # this might happen because of wrong flagged span
            # those specific flags are ignored during labeling
            raise ContextRetrievalError

        local_block.relevant = True
        return local_block

    except ContextRetrievalError:
        raise ContextRetrievalError({"message": "No local block supporting fact",
                                     "type": "supporting fact error"})
    except Exception as error:
        raise ContextRetrievalError({"message": str(error),
                                     "type": "supporting fact error"})


def get_all_context(query_name, program_content, parser, file_path, message, result_location, aux_result_df):
    """
    This is helper function to get query specific context/Block given
    program_content, start_line, end_line and name of the query.
    Args:
        name : name of the query
        program_content: Program in string format from which we need to
        extract classes
        parser : tree_sitter_parser
        file_path : file of program_content
        message : CodeQL message
        result_location: result_location protobuf with supporting fact locations
        aux_result_df: auxiliary query results dataframe
    Returns:
        A list consisting context Blocks with relevance label
    """
    result_start_line = (result_location.start_line)
    result_end_line = (result_location.end_line)
    result_start_col = (result_location.start_column)
    result_end_col = (result_location.end_column)

    try:
        # initiate context blocks list and supporting fact set
        blocks_and_metadata_list = []
        supporting_fact_block_list = []

        # get local blocks of supporting facts
        for supporting_fact in (result_location.supporting_fact_locations):
            # span with adjusted start and end point
            adjusted_supporting_fact_span = Span(supporting_fact.start_line - 1,
                                                 supporting_fact.start_column - 1,
                                                 supporting_fact.end_line - 1,
                                                 supporting_fact.end_column)
            supporting_fact_local_block = get_local_block_context(program_content,
                                                                  adjusted_supporting_fact_span)
            block_already_added = False
            for block in supporting_fact_block_list:
                if(block == supporting_fact_local_block):
                    block_already_added = True
                    break
            if not block_already_added:
                supporting_fact_block_list.append(supporting_fact_local_block)

        # codeql starts indexing from 1
        # tree-sitter starts indexing from 0
        # codeql result (a, b) -> tree-sitter span (a-1, b)
        start_line = result_start_line - 1
        start_col = result_start_col - 1
        end_line = result_end_line - 1
        end_col = result_end_col
        adjusted_result_span = Span(start_line, start_col,
                                    end_line, end_col)

        result_span_context_blocks = get_span_context(query_name, program_content,
                                                      parser, file_path,
                                                      message, adjusted_result_span,
                                                      aux_result_df)

        # add context blocks
        blocks_and_metadata_list.extend(result_span_context_blocks)

        if(query_name not in QUERIES_IGNORE_SF_ADD):
            # resolve ambiguity of relevance while adding supporting fact blocks
            for supporting_fact_block in supporting_fact_block_list:
                already_added = False
                relevance_mismatch = False
                assert supporting_fact_block.relevant
                for context_block in result_span_context_blocks:
                    if(are_same_block(context_block, supporting_fact_block)):
                        already_added = True
                        if(not context_block.relevant):  # if same block, then relevance should match
                            context_block.relevant = True
                            relevance_mismatch = True
                        break

                if(not already_added):
                    if(query_name in QUERIES_ADD_SF_AS_CONTEXT):
                        # directly add to context as similar task would
                        # be done if we context extraction seprately
                        blocks_and_metadata_list.append(supporting_fact_block)
                    else:
                        # if supporting block is not there, add them with
                        # class linkage, so that context is not discontiuous
                        _, sf_linkage_blocks = add_sf_block_with_class_heirarchy_check(program_content,
                                                                                       supporting_fact_block,
                                                                                       adjusted_result_span)
                        assert (supporting_fact_block in sf_linkage_blocks)
                        for sf_block in sf_linkage_blocks:
                            sf_already_added = False
                            for added_block in blocks_and_metadata_list:
                                if(are_same_block(added_block, sf_block)):
                                    sf_already_added = True
                                    if(not added_block.relevant):  # if same block, then relevance should match
                                        added_block.relevant = True
                                    break
                            if(not sf_already_added):
                                blocks_and_metadata_list.append(sf_block)

                        sf_dict = {'name': query_name, 'file_path': file_path,
                                   'message': message, 'start_line': result_start_line,
                                   'end_line': result_end_line, 'type': 'sf added separately',
                                   'sf_start': supporting_fact_block.start_line,
                                   'sf_end': supporting_fact_block.end_line}
                        with open('sf_block_check.log', 'a') as logger:
                            logger.write(str(sf_dict) + '\n')
                if(relevance_mismatch):
                    sf_dict = {'name': query_name, 'file_path': file_path,
                               'message': message, 'start_line': result_start_line,
                               'end_line': result_end_line, 'type': 'sf relevance mismatch',
                               'sf_start': supporting_fact_block.start_line,
                               'sf_end': supporting_fact_block.end_line}
                    with open('sf_block_check.log', 'a') as logger:
                        logger.write(str(sf_dict) + '\n')

        # deduplicate wrt to relevant
        deduplicated_blocks_and_metadata_list = []
        for i in range(len(blocks_and_metadata_list)):
            duplicate_present = False
            for j in range(len(blocks_and_metadata_list)):
                if(are_same_block(blocks_and_metadata_list[i], blocks_and_metadata_list[j])
                        and (blocks_and_metadata_list[i].relevant != blocks_and_metadata_list[j].relevant)):
                    duplicate_present = True
                    blocks_and_metadata_list[i].relevant = True
                    blocks_and_metadata_list[j].relevant = True
                elif(blocks_and_metadata_list[i] == blocks_and_metadata_list[j]):
                    duplicate_present = True

                if(duplicate_present):
                    break
            if(blocks_and_metadata_list[i] not in deduplicated_blocks_and_metadata_list):
                deduplicated_blocks_and_metadata_list.append(blocks_and_metadata_list[i])

        return deduplicated_blocks_and_metadata_list
    except ContextRetrievalError:
        raise ContextRetrievalError({"message": "Error during context block extraction",
                                     "type": "Context error"})
    except Exception as error:
        raise ContextRetrievalError({"message": str(error),
                                     "type": "Context error"})


def get_all_context_with_simplified_relevance(query_name, program_content, parser, file_path,
                                              message, result_location, aux_result_df):
    """
    This is helper function to get query specific context/Block given
    program_content, start_line, end_line and name of the query. This data
    will also contain the SIMPLIFIED relevance label.
    Args:
        name : name of the query
        program_content: Program in string format from which we need to
        extract classes
        parser : tree_sitter_parser
        file_path : file of program_content
        message : CodeQL message
        result_location: result_location protobuf with supporting fact locations
        aux_result_df: auxiliary query results dataframe
    Returns:
        A list consisting context Blocks with relevance label
    """
    context_object = BaseContexts(tree_sitter_parser)

    result_start_line = (result_location.start_line)
    result_end_line = (result_location.end_line)
    result_start_col = (result_location.start_column)
    result_end_col = (result_location.end_column)

    try:
        # initiate context blocks list and supporting fact set
        blocks_and_metadata_list = []
        supporting_fact_block_list = []

        # get local blocks of supporting facts
        for supporting_fact in (result_location.supporting_fact_locations):
            # span with adjusted start and end point
            adjusted_supporting_fact_span = Span(supporting_fact.start_line - 1,
                                                 supporting_fact.start_column - 1,
                                                 supporting_fact.end_line - 1,
                                                 supporting_fact.end_column)
            supporting_fact_local_block = get_local_block_context(program_content,
                                                                  adjusted_supporting_fact_span)
            block_already_added = False
            for block in supporting_fact_block_list:
                if(block == supporting_fact_local_block):
                    block_already_added = True
                    break
            if not block_already_added:
                supporting_fact_block_list.append(supporting_fact_local_block)

        # codeql starts indexing from 1
        # tree-sitter starts indexing from 0
        # codeql result (a, b) -> tree-sitter span (a-1, b)
        start_line = result_start_line - 1
        start_col = result_start_col - 1
        end_line = result_end_line - 1
        end_col = result_end_col
        adjusted_result_span = Span(start_line, start_col,
                                    end_line, end_col)

        result_span_context_blocks = get_span_context(query_name, program_content,
                                                      parser, file_path,
                                                      message, adjusted_result_span,
                                                      aux_result_df)
        # SIMPLIFIED relevance
        for context_block in result_span_context_blocks:
            if(not context_block.relevant):
                context_block.relevant = True

        # add context blocks
        blocks_and_metadata_list.extend(result_span_context_blocks)

        if(query_name not in QUERIES_IGNORE_SF_ADD):
            # resolve ambiguity of relevance while adding supporting fact blocks
            for supporting_fact_block in supporting_fact_block_list:
                already_added = False
                assert supporting_fact_block.relevant

                for context_block in result_span_context_blocks:
                    if(are_same_block(context_block, supporting_fact_block)):
                        already_added = True
                        break

                if(not already_added):
                    if(query_name in QUERIES_ADD_SF_AS_CONTEXT):
                        # directly add to context as similar task would
                        # be done if we context extraction seprately
                        blocks_and_metadata_list.append(supporting_fact_block)
                    else:
                        # if supporting block is not there, add them with
                        # class linkage, so that context is not discontiuous
                        _, sf_linkage_blocks = add_sf_block_with_class_heirarchy_check(program_content,
                                                                                       supporting_fact_block,
                                                                                       adjusted_result_span)
                        assert (supporting_fact_block in sf_linkage_blocks)
                        for sf_block in sf_linkage_blocks:
                            sf_already_added = False
                            # at this point all added blocks in
                            # blocks_and_metadata_list are already relevant
                            sf_block.relevant = True
                            for added_block in blocks_and_metadata_list:
                                if(are_same_block(added_block, sf_block)):
                                    sf_already_added = True
                                    break
                            if(not sf_already_added):
                                blocks_and_metadata_list.append(sf_block)

        # deduplicate wrt to relevant
        deduplicated_blocks_and_metadata_list = []
        for i in range(len(blocks_and_metadata_list)):
            duplicate_present = False
            for j in range(len(blocks_and_metadata_list)):
                if(are_same_block(blocks_and_metadata_list[i], blocks_and_metadata_list[j])
                        and (blocks_and_metadata_list[i].relevant != blocks_and_metadata_list[j].relevant)):
                    duplicate_present = True
                    blocks_and_metadata_list[i].relevant = True
                    blocks_and_metadata_list[j].relevant = True
                elif(blocks_and_metadata_list[i] == blocks_and_metadata_list[j]):
                    duplicate_present = True

                if(duplicate_present):
                    break
            if(blocks_and_metadata_list[i] not in deduplicated_blocks_and_metadata_list):
                deduplicated_blocks_and_metadata_list.append(blocks_and_metadata_list[i])

        for deduplicated_block in deduplicated_blocks_and_metadata_list:
            assert deduplicated_block.relevant
        all_blocks = context_object.get_all_blocks(program_content)
        # get non-relevant/non-context Blocks
        for block in all_blocks:
            already_in_context = False
            for context_block in deduplicated_blocks_and_metadata_list:
                if(are_same_block(block, context_block)):
                    already_in_context = True
                    break
            if(not already_in_context):
                assert (not block.relevant)
                deduplicated_blocks_and_metadata_list.append(block)

        return deduplicated_blocks_and_metadata_list
    except ContextRetrievalError:
        raise ContextRetrievalError({"message": "Error during context block extraction",
                                     "type": "Context error"})
    except Exception as error:
        raise ContextRetrievalError({"message": str(error),
                                     "type": "Context error"})
