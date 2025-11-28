from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as f


def main():
    inputPath = 'sourceData/rawFiles'
    outputPath = 'targetData'
    conf = SparkConf().set('spark.driver.memory', '2g').set('spark.executor.memory', '2g')
    spark = SparkSession.builder.appName('Retail Project').config(conf=conf).getOrCreate()
    customer_df = spark.read.option('header', 'true').csv(f'{inputPath}/customers.csv')
    customer_df = customer_df.withColumn('customer_id', f.col('customer_id').cast('int'))
    customer_df = customer_df.where(f.col('customer_id').isNotNull())
    customer_df.show()
    # customer_df.printSchema()
    order_df = spark.read.option('header', 'true').csv(f'{inputPath}/orders.csv')
    order_df = (order_df.withColumn('order_id', f.col('order_id').cast('int'))
                .withColumn('customer_id', f.col('customer_id').cast('int'))
                .withColumn('product_id', f.col('product_id').cast('int'))
                .withColumn('quantity', f.col('quantity').cast('int'))
                .withColumn('unit_price', f.col('unit_price').cast('int')))
    order_df = order_df.where((f.col('quantity') >= 0) & (f.col('order_date').isNotNull()))
    order_df.show()
    # order_df.printSchema()
    product_df = spark.read.option('header', 'true').csv(f'{inputPath}/products.csv')
    product_df = (product_df.withColumn('product_id', f.col('product_id').cast('int'))
                  .withColumn('unit_price', f.col('unit_price').cast('int'))
                  .withColumn('cost_price', f.col('cost_price').cast('int'))
                  .withColumn('active_flag', f.col('active_flag').cast('int')))
    product_df = product_df.where(f.col('product_id').isNotNull())
    product_df.show()
    # product_df.printSchema()
    # Requirement 1: Only include valid orders (Only orders with status Completed should be considered for sales
    # metrics)
    order_df = order_df.filter(f.col('status') == 'Completed')
    # Requirement 2: Enrich orders with customer information For every order, pull customer attributes:city,country,
    # segment,signup_date
    orderWithCustomerInfo = (order_df.alias('o').join(customer_df.alias('c'),
                                                      f.col('o.customer_id') == f.col('c.customer_id'), 'left')
                             .select(f.col('o.order_id'),
                                     f.col('o.order_date'),
                                     f.col('o.customer_id'),
                                     f.col('o.product_id'),
                                     f.col('o.quantity'),
                                     f.col('o.unit_price'),
                                     f.col('c.city'),
                                     f.col('c.country'),
                                     f.col('c.segment')
                                     ))
    orderWithCustomerInfo.show(5)
    # Requirement 3: Enrich orders with product information product_name,category,unit_price (from master),
    # cost_price,active_flag # Requirement 4: Handle discounts properly
    orderWithProductInfo = (order_df.alias('o').join(product_df.alias('p'),
                                                     f.col('o.product_id') == f.col('p.product_id'), 'left').
                            withColumn('discount_flag',
                                       f.coalesce(
                                           f.when(f.col('o.unit_price') < f.col('p.unit_price'), 'True').otherwise(
                                               'False')))
                            .select(f.col('o.order_id'),
                                    f.col('o.order_date'),
                                    f.col('o.customer_id'),
                                    f.col('o.product_id'),
                                    f.col('o.quantity'),
                                    f.col('o.unit_price'),
                                    f.col('p.product_name'),
                                    f.col('p.category'),
                                    f.col('p.unit_price'),
                                    f.col('p.cost_price'),
                                    f.col('p.active_flag'),
                                    f.col('discount_flag')
                                    ))
    orderWithProductInfo.show(5)
    # Requirement 8: Create a unified “fact_sales” dataset
    fact_table = (orderWithCustomerInfo.alias('oc').join(orderWithProductInfo.alias('op'),
                                                         f.col('oc.product_id') == f.col('op.product_id'), how='inner')
                  .withColumn('totalSaleAmount', f.col('oc.quantity') * f.col('oc.unit_price'))
                  .withColumn('profit', f.col('oc.quantity') * (f.col('oc.unit_price') - f.col('op.cost_price')))
                  # .withColumn('margin_percent',
                  #             ((f.col("oc.unit_price") - f.col("op.cost_price")) / f.col("oc.unit_price")) * 100)
                  ).select(
        f.col('oc.order_id'),
        f.col('oc.order_date'),
        f.col('oc.customer_id'),
        f.col('oc.city'),
        f.col('oc.segment'),
        f.col('oc.product_id'),
        f.col('op.product_name'),
        f.col('op.category'),
        f.col('op.quantity'),
        f.col('oc.unit_price').alias('selling_price'),
        f.col('op.discount_flag')
    )
    fact_table.show()
    # create csv output files from data frames out of all
    order_df.toPandas().to_csv(f'{outputPath}/order.csv', index=False, header=True)
    product_df.toPandas().to_csv(f'{outputPath}/product.csv', index=False, header=True)
    customer_df.toPandas().to_csv(f'{outputPath}/customer.csv', index=False, header=True)
    orderWithProductInfo.toPandas().to_csv(f'{outputPath}/orderWithProductInfo.csv', index=False, header=True)
    orderWithCustomerInfo.toPandas().to_csv(f'{outputPath}/orderWithCustomerInfo.csv', index=False, header=True)
    fact_table.toPandas().to_csv(f'{outputPath}/factDataTable.csv', index=False, header=True)
    print('All the data has been saved to target directory')


if __name__ == '__main__':
    main()
