examples.ip_address.data_preprocessing
======================================

.. py:module:: examples.ip_address.data_preprocessing


Attributes
----------

.. autoapisummary::

   examples.ip_address.data_preprocessing.INPUT_FILE
   examples.ip_address.data_preprocessing.OUTPUT_FILE
   examples.ip_address.data_preprocessing.processor


Classes
-------

.. autoapisummary::

   examples.ip_address.data_preprocessing.FlowDataPreprocessor


Module Contents
---------------

.. py:data:: INPUT_FILE
   :value: ''


.. py:data:: OUTPUT_FILE
   :value: 'cleaned_flows.csv'


.. py:class:: FlowDataPreprocessor(chunksize=200000)

   Preprocess large network flow CSV data in chunks.

   Handles:
   - Data cleaning and validation
   - Duplicate removal across chunks
   - Aggregation into graph-ready edge format


   .. py:attribute:: chunksize
      :value: 200000



   .. py:attribute:: required_columns
      :value: ['Flow.ID', 'Source.IP', 'Destination.IP', 'Total.Length.of.Fwd.Packets', 'Flow.Bytes.s']



   .. py:method:: clean_chunk(chunk: pandas.DataFrame) -> pandas.DataFrame

      Clean a single chunk of flow data.

      :param chunk: Raw chunk of CSV data.
      :return: Cleaned DataFrame with valid rows only.



   .. py:method:: preprocess_csv(input_csv: str, output_csv: str)

      Clean CSV in chunks and remove duplicate Flow.ID globally.

      :param input_csv: Path to raw input CSV file.
      :param output_csv: Path to save cleaned output CSV.



   .. py:method:: combine_for_graph(cleaned_csv: str, output_csv: str)

      Aggregate cleaned flow data into graph edge format.

      Groups by (Source.IP, Destination.IP) to create edges.

      :param cleaned_csv: Path to cleaned CSV file.
      :param output_csv: Path to save aggregated edge CSV.



.. py:data:: processor

